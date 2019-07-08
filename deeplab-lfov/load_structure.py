# -*-coding:utf-8-*-

"""
This script aims to investigate the structure of deeplab-lfov
"""

import tensorflow as tf
from six.moves import cPickle

# Loading net skeleton with parameters name and shapes.
with open("./net_skeleton.ckpt", "rb") as f:
    net_skeleton = cPickle.load(f)
    
    for name, shape in net_skeleton:
        print name, shape

"""
conv1_1/w (3, 3, 3, 64)
conv1_1/b (64,)
conv1_2/w (3, 3, 64, 64)
conv1_2/b (64,)
conv2_1/w (3, 3, 64, 128)
conv2_1/b (128,)
conv2_2/w (3, 3, 128, 128)
conv2_2/b (128,)
conv3_1/w (3, 3, 128, 256)
conv3_1/b (256,)
conv3_2/w (3, 3, 256, 256)
conv3_2/b (256,)
conv3_3/w (3, 3, 256, 256)
conv3_3/b (256,)
conv4_1/w (3, 3, 256, 512)
conv4_1/b (512,)
conv4_2/w (3, 3, 512, 512)
conv4_2/b (512,)
conv4_3/w (3, 3, 512, 512)
conv4_3/b (512,)
conv5_1/w (3, 3, 512, 512)
conv5_1/b (512,)
conv5_2/w (3, 3, 512, 512)
conv5_2/b (512,)
conv5_3/w (3, 3, 512, 512)
conv5_3/b (512,)
fc6/w (3, 3, 512, 1024)
fc6/b (1024,)
fc7/w (1, 1, 1024, 1024)
fc7/b (1024,)
fc8_voc12/w (1, 1, 1024, 21)
fc8_voc12/b (21,)
"""
