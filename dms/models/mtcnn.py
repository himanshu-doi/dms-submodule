'''Copyright (C) 2019 AS <parai@foxmail.com>'''

import os
import tensorflow as tf
import numpy as np
from eyegaze_prediction import SESS_FD
import eyegaze_prediction.api.dms.dms.models.align.detect_face as detect_face

from sklearn.metrics.pairwise import euclidean_distances

__all__ = ['predict']

_sess = SESS_FD

def model():
    return detect_face.create_mtcnn(_sess, None)

pnet, rnet, onet = model()
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

dir = os.path.dirname(os.path.realpath(__file__))
pb = '%s/align/PNet.pb'%(dir)
if(not os.path.exists(pb)):
    from tensorflow.python.framework import graph_util
    constant_graph = graph_util.convert_variables_to_constants(_sess, _sess.graph_def,
                                                               ['pnet/conv4-2/BiasAdd', 'pnet/prob1'])
    with tf.gfile.FastGFile(pb, mode='wb') as f:
        f.write(constant_graph.SerializeToString())
pb = '%s/align/RNet.pb'%(dir)
if(not os.path.exists(pb)):
    from tensorflow.python.framework import graph_util
    constant_graph = graph_util.convert_variables_to_constants(_sess, _sess.graph_def,
                                                               ['rnet/conv5-2/conv5-2', 'rnet/prob1'])
    with tf.gfile.FastGFile(pb, mode='wb') as f:
        f.write(constant_graph.SerializeToString())
pb = '%s/align/ONet.pb'%(dir)
if(not os.path.exists(pb)):
    from tensorflow.python.framework import graph_util
    constant_graph = graph_util.convert_variables_to_constants(_sess, _sess.graph_def,
                                                               ['onet/conv6-2/conv6-2', 'onet/conv6-3/conv6-3'])
    with tf.gfile.FastGFile(pb, mode='wb') as f:
        f.write(constant_graph.SerializeToString())


def predict(context):
    """predicts face bounding boxes on image context input"""
    frames = context['frame']
    img_sizes = context['img_sizes']
    ret = detect_face.bulk_detect_face(frames, minsize, pnet, rnet, onet, threshold, factor)
    bulk_boxes = []

    for item in ret:
        try:
            bulk_boxes.append(item[0][0])
        except TypeError:
            dummy_box = np.array([0]*5)
            bulk_boxes.append(dummy_box)
    bulk_boxes = np.array(bulk_boxes)

    if len(bulk_boxes) < 1:
        context['faces'] = []
    else:
        boxs = []
        det_b = bulk_boxes

        det_b[:,0]=np.maximum(det_b[:,0], 0)
        det_b[:,1]=np.maximum(det_b[:,1], 0)
        det_b[:,2]=np.minimum(det_b[:,2], img_sizes[:, 1])
        det_b[:,3]=np.minimum(det_b[:,3], img_sizes[:, 0])

        det_b=det_b.astype(int)
        for i in range(len(bulk_boxes)):
            boxs.append((det_b[i,0],det_b[i,1],det_b[i,2]-det_b[i,0],det_b[i,3]-det_b[i,1]))

        context['faces'] = [{'box':(x, y, w, h)} for (x, y, w, h) in boxs]
