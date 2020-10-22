''' Script for iSC/VIA/Composure/eye_contact feature. '''
# https://github.com/swook/GazeML


import sys
import glob
import tensorflow as tf
import cv2 as cv
import numpy as np
import dlib
import bz2
import shutil
from memory_profiler import profile
from config import EYE_GAZE_BATCH_SIZE
from eyegaze_prediction import GPU_AVAILABLE
from eyegaze_prediction import SESS
from eyegaze_prediction.accessories.time_profiler import timeProfile
from urllib.request import urlopen
from scipy.stats import skew

from eyegaze_prediction.api.dms.dms.models.gaze_tools import visualize_headpose_result
from eyegaze_prediction.api.dms.dms.models.gaze_tools import get_phi_theta_from_euler, limit_yaw
from eyegaze_prediction.api.dms.dms.models.gaze_tools_standalone import euler_from_matrix
from .gaze.GazeML.src.models import ELG
from .common import *


__all__ = ['get_landmarks_and_heatmaps', 'predict_eye_contact']
_landmarks_predictor = None
_data_format = 'NHWC'
sess_ = SESS


def model2():
    """
    Loading the model in the tf session
    """
    class DataSource:
        def __init__(self):
            global _data_format
            self.batch_size = EYE_GAZE_BATCH_SIZE
            self.data_format = 'NCHW' if GPU_AVAILABLE else 'NHWC'
            _data_format = self.data_format
            self.output_tensors = {
                'eye': tf.compat.v1.placeholder(tf.float32, [EYE_GAZE_BATCH_SIZE, 36, 60, 1], name='eye')
            }

        def cleanup(self):
            pass

        def create_and_start_threads(self):
            pass

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    data_source = DataSource()
    elgmodel = ELG(
        sess_, train_data={'videostream': data_source},
        first_layer_stride=1,
        num_modules=2,
        num_feature_maps=32,
        learning_schedule=[
            {
                'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']},
            },
        ],
    )

    elgmodel.initialize_if_not(training=False)
    elgmodel.checkpoint.load_all()

    eye = sess_.graph.get_tensor_by_name('eye:0')
    heatmaps = sess_.graph.get_tensor_by_name('hourglass/hg_2/after/hmap/conv/BiasAdd:0')
    landmarks = sess_.graph.get_tensor_by_name('upscale/mul:0')
    radius = sess_.graph.get_tensor_by_name('radius/out/fc/BiasAdd:0')

    return eye, heatmaps, landmarks, radius


eye, heatmaps, landmarks, radius = model2()


def draw_gaze(image_in, eye_pos, pitchyaw, length=40.0, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv.cvtColor(image_out, cv.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv.LINE_AA, tipLength=0.2)
    return image_out


def predict_eye_contact(face, gaze_heatmaps, gaze_landmarks, gaze_radius):
    """Retrieve eye landmarks, eyeball centre and iris centre --> predict eye contact"""
    # frame = context['viz_frame']
    viz_frame = 0

    can_use_eye, can_use_eyelid, can_use_iris = True, False, False
    oheatmaps, olandmarks, oradius = gaze_heatmaps, gaze_landmarks, gaze_radius
    eyes = face['eyes']
    eye_pred = []
    for j in range(2):
        eye = eyes[j]
        eye_image = eye['image']
        eye_side = eye['side']
        eye_landmarks = olandmarks[j, :]
        eye_radius = oradius[j][0]
        if eye_side == 'left':
            eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]

        eye_landmarks = np.concatenate([eye_landmarks,
                                        [[eye_landmarks[-1, 0] + eye_radius,
                                          eye_landmarks[-1, 1]]]])
        eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)),
                                           'constant', constant_values=1.0))
        eye_landmarks = (eye_landmarks *
                         eye['inv_landmarks_transform_mat'].T)[:, :2]
        eye_landmarks = np.asarray(eye_landmarks)
        iris_centre = eye_landmarks[16, :]
        eyeball_centre = eye_landmarks[17, :]
        eyeball_radius = np.linalg.norm(eye_landmarks[18, :] -
                                        eye_landmarks[17, :])
        i_x0, i_y0 = iris_centre
        e_x0, e_y0 = eyeball_centre
        theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
        phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)),
                                -1.0, 1.0))
        current_gaze = np.array([theta, phi])
        dx = -1.0 * np.sin(current_gaze[1])
        dy = -1.0 * np.sin(current_gaze[0])
        eye_pred.append([int(i_x0), int(i_y0), int(e_x0), int(e_y0), theta, phi, dx, dy])
        # visualize gaze direction
        # if can_use_eye:
        #     # Visualize landmarks
        #     cv.drawMarker(  # Eyeball centre
        #         frame, tuple(np.round(eyeball_centre).astype(np.int32)),
        #         color=(0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=4,
        #         thickness=1, line_type=cv.LINE_AA
        #     )
        #
        #     viz_frame = draw_gaze(frame, iris_centre, current_gaze,
        #                           length=120.0, thickness=1)

    return eye_pred, viz_frame


def _get_dlib_data_file(dat_name):
    dat_dir = os.path.dirname(os.path.realpath(__file__)) + '/gaze/3rdparty'
    dat_path = '%s/%s' % (dat_dir, dat_name)
    if not os.path.isdir(dat_dir):
        os.mkdir(dat_dir)

    # Download trained shape detector
    if not os.path.isfile(dat_path):
        with urlopen('http://dlib.net/files/%s.bz2' % dat_name) as response:
            with bz2.BZ2File(response) as bzf, open(dat_path, 'wb') as f:
                shutil.copyfileobj(bzf, f)

    return dat_path


def get_landmarks_predictor():
    """Get a singleton dlib face landmark predictor."""
    global _landmarks_predictor
    if not _landmarks_predictor:
        dat_path = _get_dlib_data_file('shape_predictor_68_face_landmarks.dat')
        _landmarks_predictor = dlib.shape_predictor(dat_path)
    return _landmarks_predictor


def detect_landmarks(face, context, ind):
    """ Detect 5-point facial landmarks for faces in frame. """
    six_face_marks = [33, 8, 36, 45, 48, 54]
    predictor = get_landmarks_predictor()
    l, t, w, h = face['box']
    rectangle = dlib.rectangle(left=int(l), top=int(t), right=int(l + w), bottom=int(t + h))
    landmarks_dlib = predictor(context['gray'][ind], rectangle)

    def tuple_from_dlib_shape(index):
        p = landmarks_dlib.part(index)
        return (p.x, p.y)

    num_landmarks = landmarks_dlib.num_parts
    landmarks = np.array([tuple_from_dlib_shape(i) for i in range(num_landmarks)])
    six_points = np.array([tuple_from_dlib_shape(i) for i in six_face_marks], dtype=np.float32)
    face['landmarks'] = landmarks
    face['six_points'] = six_points
    # print("landmarks_dict:", face['landmarks'][:, 0])
    skew_x = skew(face['landmarks'][:, 0], bias=False)
    skew_y = skew(face['landmarks'][:, 1], bias=False)
    return skew_x, skew_y


def detect_headpose(face, context, ind):
    """order of model points
        Tip of the nose,
        Chin,
        Left corner of the left eye,
        Right corner of the right eye,
        Left corner of the mouth,
        Right corner of the mouth
        """
    headpose_img = 0
    frame = context['frame']
    image_points = face['six_points']
    # print(image_points)
    model_points = np.array([(0.0, 0.0, 0.0),
                             (0.0, -330.0, -65.0),
                             (-225.0, 170.0, -135.0),
                             (225.0, 170.0, -135.0),
                             (-150.0, -150.0, -125.0),
                             (150.0, -150.0, -125.0)], dtype=np.float32)
    focal_length = context['gray'][ind].shape[1]
    opt_center = (context['gray'][ind].shape[1]/2, context['gray'][ind].shape[0]/2)
    camera_matrix = np.array([[focal_length, 0, opt_center[0]],
                            [0, focal_length, opt_center[1]],
                            [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)

    if not success:
        print('Not able to extract head pose for subject')

    _rotation_matrix, _ = cv.Rodrigues(rotation_vector)
    _rotation_matrix = np.matmul(_rotation_matrix, np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
    _m = np.zeros((4, 4))
    _m[:3, :3] = _rotation_matrix
    _m[3, 3] = 1
    # Go from camera space to ROS space
    _camera_to_ros = [[0.0, 0.0, 1.0, 0.0],
                      [-1.0, 0.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]]
    roll_pitch_yaw = list(euler_from_matrix(np.dot(_camera_to_ros, _m)))
    roll_pitch_yaw = limit_yaw(roll_pitch_yaw)

    phi_head, theta_head = get_phi_theta_from_euler(roll_pitch_yaw)

    # Visualize head pose
    # nose_landmark = image_points[0]
    # headpose_img = visualize_headpose_result(frame, nose_landmark, (phi_head, theta_head))
    context['viz_frame'] = headpose_img
    return phi_head, theta_head


def detect_eyes(face, context, ind):
    """ segment eye image from the landmarks found in previous steps. """
    eyes = []
    # Final output dimensions
    oh, ow = (36, 60)
    landmarks = face['landmarks']
    for corner1, corner2, is_left in [(36, 39, True), (42, 45, False)]:
        x1, y1 = landmarks[corner1, :]
        x2, y2 = landmarks[corner2, :]
        eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
        if eye_width == 0.0:
            continue
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        # Centre image on middle of eye
        translate_mat = np.asmatrix(np.eye(3))
        translate_mat[:2, 2] = [[-cx], [-cy]]
        inv_translate_mat = np.asmatrix(np.eye(3))
        inv_translate_mat[:2, 2] = -translate_mat[:2, 2]
        # Rotate to be upright
        roll = 0.0 if x1 == x2 else np.arctan((y2 - y1) / (x2 - x1))
        rotate_mat = np.asmatrix(np.eye(3))
        cos = np.cos(-roll)
        sin = np.sin(-roll)
        rotate_mat[0, 0] = cos
        rotate_mat[0, 1] = -sin
        rotate_mat[1, 0] = sin
        rotate_mat[1, 1] = cos
        inv_rotate_mat = rotate_mat.T
        # Scale
        scale = ow / eye_width
        scale_mat = np.asmatrix(np.eye(3))
        scale_mat[0, 0] = scale_mat[1, 1] = scale
        inv_scale = 1.0 / scale
        inv_scale_mat = np.asmatrix(np.eye(3))
        inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale
        # Centre image
        centre_mat = np.asmatrix(np.eye(3))
        centre_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
        inv_centre_mat = np.asmatrix(np.eye(3))
        inv_centre_mat[:2, 2] = -centre_mat[:2, 2]
        # Get rotated and scaled, and segmented image
        transform_mat = centre_mat * scale_mat * rotate_mat * translate_mat
        inv_transform_mat = (inv_translate_mat * inv_rotate_mat * inv_scale_mat *
                             inv_centre_mat)
        eye_image = cv.warpAffine(context['gray'][ind], transform_mat[:2, :], (ow, oh))
        if is_left:
            eye_image = np.fliplr(eye_image)
        eyes.append({
            'image': eye_image,
            'inv_landmarks_transform_mat': inv_transform_mat,
            'side': 'left' if is_left else 'right',
        })
    face['eyes'] = eyes


def eye_preprocess(eye):
    eye = cv.equalizeHist(eye)
    eye = eye.astype(np.float32)
    eye *= 2.0 / 255.0
    eye -= 1.0
    eye = np.expand_dims(eye, -1 if _data_format == 'NHWC' else 0)
    return eye


# @profile
# @timeProfile(lines_to_print=30, strip_dirs=True)
def get_landmarks_and_heatmaps(context):
    no_face = []
    no_eyes = []
    phi_head = []
    theta_head = []
    skew_x = []
    skew_y = []
    eye_batch = []
    if not context['faces']:
        print("[!] No face detected in images")
        return phi_head, theta_head, skew_x, skew_y, no_face, no_eyes
    else:
        dummy_eye = np.zeros((2, 36, 60, 1))
        for ind, face in enumerate(context['faces']):
            if sum(face['box']) == 0:
                no_face.append(True)
                no_eyes.append(True)
                skew_x.append(0)
                skew_y.append(0)
                phi_head.append(0)
                theta_head.append(0)
                eye_batch.append(dummy_eye)
                continue
            no_face.append(False)
            # x, y, w, h = face['box']
            # if (w < 160) or (h < 160):
            #     print("width height issue")
            #     continue
            # detect facial landmarks
            _skew_x, _skew_y = detect_landmarks(face, context, ind)
            skew_x.append(_skew_x)
            skew_y.append(_skew_y)

            # detect eye using landmarks
            detect_eyes(face, context, ind)

            # detect head pose using landmarks
            _phi_head, _theta_head = detect_headpose(face, context, ind)
            phi_head.append(_phi_head)
            theta_head.append(_theta_head)
            if len(face['eyes']) != 2:
                no_eyes.append(True)
                eye_batch.append(dummy_eye)
                continue
            no_eyes.append(False)
            eye1 = eye_preprocess(face['eyes'][0]['image'])
            eye2 = eye_preprocess(face['eyes'][1]['image'])
            eyeI = np.concatenate((eye1, eye2), axis=0)
            eyeI = eyeI.reshape(2, 36, 60, 1)  # 36,60,18 = 77760
            eye_batch.append(eyeI)
        eye_batch = np.array(eye_batch).reshape(EYE_GAZE_BATCH_SIZE, 36, 60, 1)
        placeholder_1 = sess_.graph.get_tensor_by_name('learning_params/Placeholder_1:0')
        feed_dict = {eye: eye_batch, placeholder_1: False}
        oheatmaps, olandmarks, oradius = sess_.run((heatmaps, landmarks, radius), feed_dict=feed_dict)
        context['gazes'] = (oheatmaps, olandmarks, oradius)
        # print('!!!!!!! batch of 5 processed !!!!!!!!!')
        return phi_head, theta_head, skew_x, skew_y, no_face, no_eyes
