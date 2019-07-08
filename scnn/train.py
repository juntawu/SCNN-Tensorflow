"""
Training script for the SCNN network on the Changshu & Tusimple dataset for lane detction (semantic image segmentation).

This script trains the model using *** dataset,
which contains approximately *** images for training and *** images for validation.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import pdb

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from create_model.scnn_model import SCNNModel
from load_data.image_reader import ImageReader
from load_data.utils import decode_labels



BATCH_SIZE = 4
EPOCH = 20
LEARNING_RATE = 1e-4
NUM_STEPS = 100000
DATA_DIRECTORY = './dataset/'
DATA_LIST_PATH = './dataset/data_list/train_gt.txt'
INPUT_SIZE = '512,640'
MEAN_IMG = tf.Variable(np.array((104.00698793,116.66876762,122.67891434)), trainable=False, dtype=tf.float32)
RANDOM_SCALE = True
PRETRAINED_PARAMS_PATH = './pretrained_params/pretrained_params.ckpt'
SAVE_DIR = './prediction/'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = './snapshots/'
WEIGHTS_PATH   = None

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


def get_arguments():
    """
    Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="SCNN Network")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--epoch", type=int, default = EPOCH,
                        help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for training.")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--data_list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--input_size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--pretrained_params_path", type=str, default=PRETRAINED_PARAMS_PATH,
                        help="Where restore model parameters from.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save figures with predictions.")
    parser.add_argument("--save_num_images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save_pred_every", type=int, default=SAVE_PRED_EVERY,
                        help="Save figure with predictions and ground truth every often.")
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")

    return parser.parse_args()



def save(saver, sess, snapshot_dir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(snapshot_dir, model_name)

    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')



def main():
    args = get_arguments()
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)



    # Create queue coordinator.
    coord = tf.train.Coordinator()

    ## load image_batch and label_batch
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            input_size,
            RANDOM_SCALE,
            coord,
            args.epoch)
        image_batch, label_batch1, label_batch2 = reader.dequeue(args.batch_size)

    dataset_size = reader.dataset_size


    ## define SCNN network and loss
    scnn = SCNNModel(args.pretrained_params_path)
    loss1, loss2, loss = scnn.loss(image_batch, label_batch1, label_batch2)

    # debug
    #loss1, loss2, loss, prediction1, gt1, prediction2, gt2 = scnn.loss(image_batch, label_batch1, label_batch2)
    
    

    trainable = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss, var_list=trainable)

    # pred = scnn.preds(image_batch)

    tf.summary.scalar('total loss', loss)
    tf.summary.scalar('loss1', loss1)
    tf.summary.scalar('loss2', loss2)

    merged_summary_op = tf.summary.merge_all()


    ## set up tensorflow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    ## initialize variables
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer() )
    sess.run(init)
    
    # inspect weights
    #scnn_d_w = [var for var in tf.global_variables() if 'spatial_conv_D' in var.name]
    #conv1_1_w = [var for var in tf.global_variables() if 'conv1_1/w' in var.name]
    #conv4_3_b = [var for var in tf.global_variables() if 'conv4_3/b' in var.name]
    #conv4_3_w = [var for var in tf.global_variables() if 'conv4_3/w' in var.name]
    #conv4_3_w = [var.name for var in tf.global_variables() if 'conv4_3/w' in var.name]
    #pdb.set_trace()

    # store graph
    summary_writer = tf.summary.FileWriter('./summary', graph=tf.get_default_graph())

    # Saver for storing checkpoints of the model.
    # max_to_keep indicates the maximum number of last checkpoint models to store
    #saver = tf.train.Saver(var_list=trainable, max_to_keep=40)
    saver = tf.train.Saver(max_to_keep=40)
    
    # inspect trainable weights
    #trainable_weights = [var.name for var in tf.trainable_variables() ]
    #pdb.set_trace()

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    # OutofRangeError will be thrown when the queue is empty
    try:
        ## Start training iteration
        for step in range(args.num_steps):
            start_time = time.time()
            
            #prediction1_value, gt1_value, prediction2_value, gt2_value, loss1_value = sess.run([prediction1, gt1, prediction2, gt2, loss1])
            #pdb.set_trace()

            if step % args.save_pred_every == 0:
                #loss_value, images, labels, preds, _ = sess.run([loss, image_batch, label_batch1, label_batch2, pred, optimizer])
                # loss1_value, loss2_value, loss_value, \
                # images, labels1, label2, preds, summary, _ = \
                #     sess.run([loss1, loss2, loss,
                #               image_batch, label_batch1, label_batch2, pred, merged_summary_op, optimizer])

                loss1_value, loss2_value, loss_value, \
                images, labels1, label2, summary, _ = \
                    sess.run([loss1, loss2, loss,
                              image_batch, label_batch1, label_batch2, merged_summary_op, optimizer])

                # fig, axes = plt.subplots(args.save_num_images, 3, figsize=(16, 12))
                # for i in xrange(args.save_num_images):
                #     axes.flat[i * 3].set_title('data')
                #     # axes.flat[i * 3].imshow((images[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8))
                #     axes.flat[i * 3].imshow((images[i])[:, :, ::-1].astype(np.uint8))
                #
                #     axes.flat[i * 3 + 1].set_title('mask')
                #     axes.flat[i * 3 + 1].imshow(decode_labels(labels1[i, :, :, 0]))
                #
                #     axes.flat[i * 3 + 2].set_title('pred')
                #     axes.flat[i * 3 + 2].imshow(decode_labels(preds[i, :, :, 0]))
                # plt.savefig(args.save_dir + str(start_time) + ".png")
                # plt.close(fig)


                save(saver, sess, args.snapshot_dir, step)

            else:
                loss1_value, loss2_value, loss_value, summary, _ = \
                    sess.run([loss1, loss2, loss, merged_summary_op, optimizer])

            #prediction1_value, gt1_value, prediction2_value, gt2_value = sess.run([prediction1, gt1, prediction2, gt2])
            #pdb.set_trace()

            # record loss
            summary_writer.add_summary(summary, step)

            duration = time.time() - start_time
            print('[Epoch {:d}] ({:d} / {:d})'.format( (step+1)*args.batch_size/dataset_size, (step+1)*args.batch_size%dataset_size, dataset_size) )
            print('Step {:d}: \t loss1 = {:.3f}, loss2 = {:.3f} ({:.3f} sec/step)'.format(step, loss1_value, loss2_value, duration))

    except tf.errors.OutOfRangeError:
        print('Done(OutofRangeError)! Now kill all threads')

    coord.request_stop()
    coord.join(threads)
    print('All threads are stopped')


if __name__ == '__main__':
    main()
