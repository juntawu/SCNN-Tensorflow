"""
Run SCNN model on a given image.

"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

from PIL import Image
import cv2

import tensorflow as tf
import numpy as np

from create_model.scnn_model import SCNNModel
from load_data import  ImageReader, decode_labels

SAVE_DIR = './output/'
# IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
# IMG_MEAN = np.array([0.3598, 0.3653, 0.3662]) * 256

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="SCNN Network Inference.")
    parser.add_argument("img_path", type=str,
                        help="Path to the RGB image file.")
    parser.add_argument("model_weights", type=str,
                        help="Path to the file with model weights.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    return parser.parse_args()


def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    print(ckpt_path)
    saver.restore(sess, ckpt_path)
    # saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
    print("Restored model parameters from {}".format(ckpt_path))


def main():
    """Create the model and start the evaluation process."""
    # args = get_arguments()

    # Prepare image.
    img_path = '/home/lishanliao/my_work/SCNN/data/CULane/TSD-Lane-00052/TSD-Lane-00052-00081.png'
    label_path = '/home/lishanliao/my_work/SCNN/data/CULane/label/TSD-Lane-00052/TSD-Lane-00052-00081.png'
    img = tf.image.decode_jpeg(tf.read_file(img_path), channels=3)
    label = tf.image.decode_png(tf.read_file(label_path), channels=1)
    # Convert RGB to BGR.
    # img_r, img_g, img_b = tf.split(split_dim=2, num_split=3, value=img)
    img_r, img_g, img_b = tf.split(value=img, num_or_size_splits=3, axis=2)
    # img = tf.cast(tf.concat(2, [img_b, img_g, img_r]), dtype=tf.float32)
    img = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)
    # Extract mean.
    # img -= IMG_MEAN
    img = tf.image.resize_images(img, [512, 640])
    label = tf.image.resize_images(label, [512, 640])

    # Create network.
    path_pretrained_params = './pretrained_params/pretrained_params.ckpt'
    net = SCNNModel(path_pretrained_params)

    # Predictions.
    pred, exist = net.preds(tf.expand_dims(img, dim=0))

    # Set up TF session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.initialize_all_variables()

    sess.run(init)

    # Perform inference.
    image, label_image, conv_feature, scnn_feature, preds, exists \
        = sess.run([tf.cast(img, tf.uint8),
                    tf.cast(label, tf.uint8),
                    tf.cast(net.fc7, tf.uint8),
                    tf.cast(net.scnn, tf.uint8),
                    pred, exist])

    print (image.shape, type(image))
    print ("Probability of existence: ", exists)
    msk = decode_labels(np.array(preds)[0, :, :, 0])
    mask1 = Image.fromarray(msk)
    mask2 = np.array(preds[0]) * 50

    # cv2.imshow('conv', np.array(conv_feature[0, :, :, 0]))
    # cv2.imshow('scnn', np.array(scnn_feature[0, :, :, 0]))

    cv2.imshow('image', np.array(image))
    cv2.imshow('label', np.array(label_image)*50)
    # cv2.imshow('mask1', np.array(mask1))
    # cv2.imshow('mask2', mask2)
    cv2.waitKey(0)
    
    #writer = tf.summary.FileWriter('./models/', sess.graph)


if __name__ == '__main__':
    main()