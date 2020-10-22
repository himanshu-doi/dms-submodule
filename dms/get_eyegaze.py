#!/usr/bin/env python3
""" iSC/VIA/Composure/eye_contact main script. """
import os
import cv2
import numpy as np
import pandas as pd
# from memory_profiler import profile
from config import DETECT
from config import EYE1_FEATS
from config import EYE2_FEATS
from config import HEAD_FEATS
from config import ML_THRESH
from eyegaze_prediction import GAZE_MODEL
from eyegaze_prediction import BusinessException
from eyegaze_prediction.accessories.time_profiler import timeProfile
from eyegaze_prediction.api.dms.dms.models.gaze_inference import get_landmarks_and_heatmaps
from eyegaze_prediction.api.dms.dms.models.gaze_inference import predict_eye_contact
from .models.mtcnn import predict as face_detect


def inference_gaze_df(test_df):
    predicted_class = GAZE_MODEL.predict(test_df)
    model_confidence_list = GAZE_MODEL.predict_proba(test_df)[:, 1]
    return predicted_class, model_confidence_list


# Writing Text on image
def put_text(frame, text_string):
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    position = (50, 50)
    fontScale = 2
    fontColor = (0, 0, 255)
    thickness = 3
    frame = cv2.putText(frame, text_string,
                position,
                font,
                fontScale,
                fontColor,
                thickness,
                cv2.LINE_AA)
    return frame


def visualize_output(context, viz_frame, filepath, pred):
    # frame = context['frame']
    for face in context['faces']:
        x, y, w, h = face['box']
        cv2.rectangle(viz_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    viz_frame = put_text(viz_frame, "outside_gaze: {}".format(str(pred)))
    output_path = os.path.split(filepath)[1][:-4] + "_" + DETECT[0] + ".png"
    print(output_path)
    cv2.imwrite('./gazeml_outputs/{}'.format(output_path), viz_frame)


def resize(frame):
    resized = cv2.resize(frame, (1024, int(1024 * frame.shape[0] / frame.shape[1])), cv2.INTER_LINEAR)
    return resized


# @profile
# @timeProfile(lines_to_print=30, strip_dirs=False)
def predict_eyegaze(file_path):
    frames = []
    bws = []
    img_sizes = []
    test_df = pd.DataFrame(columns=HEAD_FEATS + EYE1_FEATS + EYE2_FEATS)
    dummy_data = pd.DataFrame(np.zeros((1, 21)), columns=HEAD_FEATS+EYE1_FEATS+EYE2_FEATS)
    for f_name in file_path:
        if not os.path.isfile(f_name):
            raise BusinessException("Invalid file path")
        resized = resize(np.array(cv2.imread(f_name)))
        img_sizes.append(np.array(resized.shape)[0:2])
        frames.append(resized)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        bws.append(gray)
    frames = np.array(frames)
    bws = np.array(bws)
    img_sizes = np.array(img_sizes)
    context = {'frame': frames, 'gray': bws, 'img_sizes': img_sizes}

    # Running MTCNN fd
    face_detect(context)

    # Head pose, face landmark skew and pupil landmark estimation
    phi_head, theta_head, skew_x, skew_y, no_face, no_eyes = get_landmarks_and_heatmaps(context)
    oheatmaps, olandmarks, oradius = context['gazes']
    for i in range(len(file_path)):
        if no_face[i] or no_eyes[i]:
            test_df = test_df.append(dummy_data, ignore_index=True)
            continue
        face = context['faces'][i]
        gaze_heatmaps = oheatmaps[i*2: (i*2)+2]
        gaze_landmarks = olandmarks[i*2: (i*2)+2]
        gaze_radius = oradius[i*2: (i*2)+2]

        # Running eye gaze estimator
        eye_pred, viz_frame = predict_eye_contact(face, gaze_heatmaps, gaze_landmarks, gaze_radius)

        try:
            bbox = face['box']
            bbox_area = float(bbox[2] * bbox[3])
        except TypeError as e:
            bbox_area = 0

        eye1 = pd.DataFrame(np.array([eye_pred[0]]), columns=EYE1_FEATS)
        eye2 = pd.DataFrame(np.array([eye_pred[1]]), columns=EYE2_FEATS)
        head = pd.DataFrame(np.array([[bbox_area, phi_head[i], theta_head[i], skew_x[i], skew_y[i]]]),
                            columns=HEAD_FEATS)
        item_df = pd.concat([head, eye1, eye2], axis=1)
        test_df = test_df.append(item_df, ignore_index=True)

    predicted_class, model_confidence_list = inference_gaze_df(test_df)
    # print(model_confidence_list)
    predictions = np.array([1 if prob > ML_THRESH else 0 for prob in model_confidence_list])
    predictions[no_eyes] = -1
    return list(map(str, predictions))
