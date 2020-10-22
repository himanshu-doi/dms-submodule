#!/usr/bin/env python3
""" iSC/VIA/Composure/eye_contact main script. """
import glob
import os
from pprint import pprint
import pandas as pd
import cv2
# import joblib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v', dest='video', default=0,
                    help='specify video source, e.g: http://192.168.1.101:4747/video')
parser.add_argument('-d', dest='detect', default='cv2',
                    help='which method to detect location of face, cv2 or mtcnn')
parser.add_argument('-f', dest='filter', default='1',
                    help='sample rate of the input')
parser.add_argument('-p', '--path', dest='path', default='1', type=str,
                    help='specify video source')

networks = ['gaze']
parser.add_argument('-n', '--network',
                    default=networks,
                    help='networks to be enabled from %s, default all is enabled' % (networks),
                    type=str, nargs='+')
args = parser.parse_args()
print('networks: %s' % (args.network))
cv2_root = os.path.dirname(os.path.realpath(cv2.__file__))

if ('gaze' in args.network):
    from models.gaze_inference import get_landmarks_and_heatmaps
    from models.gaze_inference import predict_eye_contact

if (args.detect == 'cv2'):
    haar_face_cascade = cv2.CascadeClassifier('%s/data/haarcascade_frontalface_alt.xml' % (cv2_root))


    def face_detect(context):
        frame = context['frame']
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxs = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        context['faces'] = [{'box': (x, y, w, h)} for (x, y, w, h) in boxs]
        colorList = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
        i=0
        for (x, y, w, h) in boxs:
            cv2.rectangle(context['frame'], (x, y), (x + w, y + h), colorList[i], 2)
            i = i + 1
else:
    from .models.mtcnn import predict as face_detect


def frames_read_main():
    output_path = "/home/himanshu/Downloads/dms_output_csvs/"
    input_path = '/home/himanshu/Downloads/Frames_HM/'
    for vid_name in os.listdir(input_path):
        pred_df = pd.DataFrame(columns=['frame_id'])
        print(vid_name, len(os.listdir(os.path.join(input_path, vid_name))))
        for ind, img in enumerate(os.listdir(os.path.join(input_path, vid_name))):
    # dic_gaze = {}
    # path_frames = args.path
    # for imgs in glob.glob(path_frames + '*.png'):
            iris_x, iris_y, eyeball_x, eyeball_y, theta, phi,eye_contact,blink = 0,0,0,0,0,0,0,False
            print("Frame number ----> {} ".format(img))
            # split = imgs.split('/')
            # file_name = '/home/vineetsingh/Pictures/eye_test_img/' + split[-2] + ' ' + split[-1] + '.png'
            frame = cv2.imread(os.path.join(input_path, vid_name, img))
            height, width, _ = frame.shape
            print(height,width)
            frame = cv2.resize(frame, (1024, int(1024 * height / width)), cv2.INTER_LINEAR)
            context = {'frame': frame, 'gray': cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)}
            face_detect(context)
            if 'gaze' in args.network : get_landmarks_and_heatmaps(context)
            iris_x, iris_y, eyeball_x, eyeball_y, theta, phi,highly_confident = predict_eye_contact(context)
            print("image: {0}/n{1}".format(img, [iris_x, iris_y, eyeball_x, eyeball_y, theta, phi, highly_confident, eye_contact]))
        # cv2.imshow('Window', frame)
        # key = cv2.waitKey(5000) & 0xFF
        # if key == ord("y"):
        #     eye_contact = 1
        # if key == ord("n"):
        #     eye_contact = 0
        # if key == ord("o"):
        #     eye_contact = 'out'
        # print(eye_contact,highly_confident)
        # dic_gaze[imgs] = [iris_x, iris_y, eyeball_x, eyeball_y, theta, phi, highly_confident, eye_contact]
    #     dic_gaze[imgs] = [iris_x, iris_y, eyeball_x, eyeball_y, theta, phi, highly_confident]
    # joblib.dump(dic_gaze, '/home/vineetsingh/Downloads/eye-contact-vineet-data16.pkl')
    # cv2.destroyAllWindows()
    #         pred_df.at[ind, 'frame_id'] = img
    #         pred_df.at[ind, 'iris_x'] = iris_x
    #         pred_df.at[ind, 'iris_y'] = iris_y
    #         pred_df.at[ind, 'eyeball_x'] = eyeball_x
    #         pred_df.at[ind, 'eyeball_y'] = eyeball_y
    #         pred_df.at[ind, 'theta'] = theta
    #         pred_df.at[ind, 'phi'] = phi
    #         pred_df.at[ind, 'highly_confident'] = highly_confident
    #
    #     pred_df.to_csv(os.path.join(output_path, '{}_dms.csv'.format(vid_name)))

if __name__ == '__main__':
    frames_read_main()

