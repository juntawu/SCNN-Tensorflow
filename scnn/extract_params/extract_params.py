"""Run DeepLab-LargeFOV on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

from PIL import Image

import tensorflow as tf
import numpy as np

from deeplab_model import DeepLabLFOVModel

SAVE_DIR = '../pretrained_params/pretrained_params.ckpt'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
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
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    # Create network.
    net = DeepLabLFOVModel()

    # Which variables to load.
    trainable = tf.trainable_variables()

      
    # Set up TF session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.initialize_all_variables()
    
    sess.run(init)
    
    # Load weights.
    saver = tf.train.Saver(var_list=trainable)
    load(saver, sess, args.model_weights)

    path_to_save_parmas = '../pretrained_params/'
    net.save_convs_params(sess, path_to_save_parmas)
    #writer = tf.summary.FileWriter('./deeplab_graph/', sess.graph)

    
if __name__ == '__main__':
    main()
