"""
Run SCNN model on a given image.

"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import pdb

from PIL import Image
import cv2

import tensorflow as tf
import numpy as np

from create_model.scnn_model import SCNNModel
from load_data import  ImageReader, decode_labels

SAVE_DIR = './output/'
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
# IMG_MEAN = np.array([0.3598, 0.3653, 0.3662]) * 256

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="SCNN Network Inference.")
    parser.add_argument("--img_path", type=str,
                        help="Path to the RGB image file.")
    parser.add_argument("--model_weights", type=str,
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
    args = get_arguments()

    # Prepare image.
    img = tf.image.decode_jpeg(tf.read_file(args.img_path), channels=3)
    # Convert RGB to BGR.
    # img_r, img_g, img_b = tf.split(split_dim=2, num_split=3, value=img)
    img_r, img_g, img_b = tf.split(value=img, num_or_size_splits=3, axis=2)
    # img = tf.cast(tf.concat(2, [img_b, img_g, img_r]), dtype=tf.float32)
    img = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN
    img = tf.image.resize_images(img, [512, 640])

    # Create network.
    net = SCNNModel()

    # Predictions.
    pred, exist = net.preds(tf.expand_dims(img, dim=0))

    # Set up TF session and initialize variables.
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.initialize_all_variables()

    sess.run(init)

    # Inspect weights
    scnn_d_w = [var for var in tf.global_variables() if 'spatial_conv_D' in var.name]
    conv4_3_w = [var for var in tf.global_variables() if 'conv4_3/w' in var.name]
    #pdb.set_trace()

    # Load weights.
    saver = tf.train.Saver()
    load(saver, sess, args.model_weights)

    # inspect trainable weights
    trainable_weights = [var.name for var in tf.trainable_variables() ]
    #pdb.set_trace()


    # Perform inference.
    image, scnn_feature, preds, exists = sess.run([tf.cast(img, tf.uint8), net.scnn, pred, exist])


    print ("Probability of existence: ", exists)
    msk = decode_labels(np.array(preds)[0, :, :, 0])
    im1 = Image.fromarray(msk)
    im2 = np.array(preds[0]) * 50
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    im1.save(args.save_dir + 'mask.png')
    # cv2.imshow('scnn', np.array(scnn_feature[0, :, :, 0]))
    #cv2.imshow('image', image)
    #cv2.imshow('mask', im2)
    #cv2.waitKey(0)

    print('The output file has been saved to {}'.format(args.save_dir + 'mask.png'))

    #writer = tf.summary.FileWriter('./models/', sess.graph)


if __name__ == '__main__':
    main()
