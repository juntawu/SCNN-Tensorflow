# -*-coding:utf-8-*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


KW = 9  ## width of SCNN kernel
DEPTH = 128

def create_scnn(features):
    with tf.name_scope("SCNN_D") as scope:
        # with slim.arg_scope([slim.conv2d], weights_regularizer=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.01), reuse=True):
        with slim.arg_scope([slim.conv2d], weights_regularizer=None,
                            #weights_initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2.0 /DEPTH*DEPTH*KW*5)),
                            weights_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                            reuse=True):
            B = features.shape.as_list()[0]
            H = features.shape.as_list()[1]
            W = features.shape.as_list()[2]
            C = features.shape.as_list()[3]

            # features 欲进行处理的特征层
            batch_size = B  # batch 大小
            slice_d = []
            
            # 特征图的高度为 features.shape.as_list()[1]
            # slice: B * 1 * W * C
            for i in range(H): #SCNN_D
                j = i + 1 # 下一个 slice
                slice_d.append( tf.strided_slice(features, [0,i,0,0], [B, j, W, C], strides=[1,1,1,1]) )
            
            
            ## slice_d 中元素的维度都是[B * 1 * W * C]
            slice_sum = slice_d[0]
            slice_concat = [slice_d[0]]
             
            for i in range(H-1):
                if 0==i:
                    reuse = None
                else:
                    reuse = True
                slice_sum = slim.conv2d(slice_sum, C, kernel_size=[1, KW], reuse = reuse, rate=1, activation_fn=tf.nn.relu, normalizer_fn=None, scope='spatial_conv_D') + slice_d[i+1]
                slice_concat.append(slice_sum)

            features_D = tf.concat(slice_concat, 1)
            print 'features_D shape:', features_D.shape

    with tf.name_scope("SCNN_U") as scope:
        #with slim.arg_scope([slim.conv2d], weights_regularizer=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.01), reuse=True):
        with slim.arg_scope([slim.conv2d], weights_regularizer=None,
                            weights_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                            reuse=True):
            B = features_D.shape.as_list()[0]
            H = features_D.shape.as_list()[1]
            W = features_D.shape.as_list()[2]
            C = features_D.shape.as_list()[3]

            # features 欲进行处理的特征层
            batch_size = B  # batch 大小
            slice_d = []

            # 特征图的高度为 features.shape.as_list()[1]
            # slice: B * 1 * W * C
            for i in range(H):  # SCNN_D
                j = i + 1  # 下一个 slice
                slice_d.append(tf.strided_slice(features_D, [0, i, 0, 0], [B, j, W, C], strides=[1, 1, 1, 1]))

            ## slice_d 中元素的维度都是[B * W * C]
            slice_sum = slice_d[H-1]
            slice_concat = [slice_d[H-1]]

            for i in range(H - 1, 0, -1):
                if H-1 == i:
                    reuse = None
                else:
                    reuse = True
                slice_sum = slim.conv2d(slice_sum, C, kernel_size=[1, 9], reuse=reuse, rate=1, activation_fn=tf.nn.relu, normalizer_fn=None, scope='spatial_conv_U') + slice_d[i - 1]
                slice_concat.insert(0, slice_sum)

            features_U = tf.concat(slice_concat, 1)
            print 'features_U shape:', features_U.shape


    with tf.name_scope("SCNN_R") as scope:
        #with slim.arg_scope([slim.conv2d], weights_regularizer=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.01), reuse=True):
        with slim.arg_scope([slim.conv2d], weights_regularizer=None,
                            weights_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                            reuse=True):

            B = features_U.shape.as_list()[0]
            H = features_U.shape.as_list()[1]
            W = features_U.shape.as_list()[2]
            C = features_U.shape.as_list()[3]

            # features 欲进行处理的特征层
            batch_size = B  # batch 大小
            slice_d = []

            # 特征图的高度为 features.shape.as_list()[1]
            # slice: B * H * 1 * C
            for i in range(W):  # SCNN_D
                j = i + 1  # 下一个 slice
                slice_d.append(tf.strided_slice(features_U, [0, 0, i, 0], [B, H, j, C], strides=[1, 1, 1, 1]))

            ## slice_d 中元素的维度都是[B * H * 1 * C]
            slice_sum = slice_d[0]
            slice_concat = [slice_d[0]]

            for i in range(W - 1):
                if 0 == i:
                    reuse = None
                else:
                    reuse = True
                slice_sum = slim.conv2d(slice_sum, C, kernel_size=[KW, 1], reuse=reuse, rate=1, activation_fn=tf.nn.relu,
                                        normalizer_fn=None, scope='spatial_conv_R') + slice_d[i + 1]
                slice_concat.append(slice_sum)

            features_R = tf.concat(slice_concat, 2)
            print 'features_R shape:', features_R.shape


    with tf.name_scope("SCNN_L") as scope:
        #with slim.arg_scope([slim.conv2d], weights_regularizer=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.01), reuse=True):
        with slim.arg_scope([slim.conv2d], weights_regularizer=None,
                            weights_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                            reuse=True):

            B = features_R.shape.as_list()[0]
            H = features_R.shape.as_list()[1]
            W = features_R.shape.as_list()[2]
            C = features_R.shape.as_list()[3]

            # features 欲进行处理的特征层
            batch_size = B  # batch 大小
            slice_d = []

            # 特征图的高度为 features.shape.as_list()[1]
            # slice: B * H * 1 * C
            for i in range(W):  # SCNN_D
                j = i + 1  # 下一个 slice
                slice_d.append(tf.strided_slice(features_R, [0, 0, i, 0], [B, H, j, C], strides=[1, 1, 1, 1]))

            ## slice_d 中元素的维度都是[B * W * C]
            slice_sum = slice_d[W-1]
            slice_concat = [slice_d[W-1]]

            for i in range(W - 1, 0, -1):
                if W-1 == i:
                    reuse = None
                else:
                    reuse = True
                slice_sum = slim.conv2d(slice_sum, C, kernel_size=[KW, 1], reuse=reuse, rate=1, activation_fn=tf.nn.relu,
                                        normalizer_fn=None, scope='spatial_conv_L') + slice_d[i - 1]
                slice_concat.append(slice_sum)

            features_L = tf.concat(slice_concat, 2)
            print 'features_L shape:', features_L.shape

    return features_L
