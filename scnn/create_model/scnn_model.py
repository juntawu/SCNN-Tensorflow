import tensorflow as tf
from six.moves import cPickle
from create_scnn import create_scnn

import pdb

# Loading net skeleton with parameters name and shapes.
with open("./create_model/net_skeleton.ckpt", "rb") as f:
# with open("./net_skeleton.ckpt", "rb") as f:
    net_skeleton = cPickle.load(f)

# The DeepLab-LargeFOV model can be represented as follows:
## input -> [conv-relu](dilation=1, channels=64) x 2 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=128) x 2 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=256) x 3 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=512) x 3 -> [max_pool](stride=1)
##       -> [conv-relu](dilation=2, channels=512) x 3 -> [max_pool](stride=1) -> [avg_pool](stride=1)
##       -> [conv-relu](dilation=12, channels=1024) -> [dropout]
##       -> [conv-relu](dilation=1, channels=1024) -> [dropout]
##       -> [conv-relu](dilation=1, channels=21) -> [pixel-wise softmax loss].
num_layers    = [2, 2, 3, 3, 3, 1, 1, 1]
dilations     = [[1, 1],
                 [1, 1],
                 [1, 1, 1],
                 [1, 1, 1],
                 [2, 2, 2],
                 [12], 
                 [1], 
                 [1]]
n_classes = 5
# All convolutional and pooling operations are applied using kernels of size 3x3; 
# padding is added so that the output of the same size as the input.
ks = 3
DROPOUT = 0.5

def create_variable(name, shape):
    """Create a convolution filter variable of the given name and shape,
       and initialise it using Xavier initialisation 
       (http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf).
    """
    initialiser = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
    variable = tf.Variable(initialiser(shape=shape), name=name)
    return variable

def create_bias_variable(name, shape):
    """Create a bias variable of the given name and shape,
       and initialise it to zero.
    """
    initialiser = tf.constant_initializer(value=0.0, dtype=tf.float32)
    variable = tf.Variable(initialiser(shape=shape), name=name)
    return variable

class SCNNModel(object):
    
    def __init__(self, weights_path=None):
        """Create the model.
        
        Args:
          weights_path: the path to the cpkt file with dictionary of weights from .caffemodel.
        """
        self.variables = self._create_variables(weights_path)
        #input = tf.placeholder(tf.float32, [4, 512, 640, 3], name='input')
        #self._create_network(input)

        
    def _create_variables(self, weights_path):
        """Create all variables used by the network.
        This allows to share them between multiple calls 
        to the loss function.
        
        Args:
          weights_path: the path to the ckpt file with dictionary of weights from .caffemodel. 
                        If none, initialise all variables randomly.
        
        Returns:
          A dictionary with all variables.
        """
        var = list()
        index = 0
        
        if weights_path is not None:
            with open(weights_path, "rb") as f:
                weights = cPickle.load(f) # Load pre-trained weights.
                #print(weights['conv1_1/w'])
                for name, shape in net_skeleton:
                    if 'conv' in name:  ## only load convolution weights
                        var.append(tf.Variable(weights[name],
                                               name=name))
                del weights
        else:
            # Initialise all weights randomly with the Xavier scheme,
            # and 
            # all biases to 0's.
            for name, shape in net_skeleton:
                if 'conv' in name:  ## only load convolution weights
                    if "/w" in name: # Weight filter.
                        w = create_variable(name, list(shape))
                        var.append(w)
                    else:
                        b = create_bias_variable(name, list(shape))
                        var.append(b)
        return var
    
    
    def _create_network(self, input_batch, keep_prob):
        """Construct DeepLab-LargeFOV network.
        
        Args:
          input_batch: batch of pre-processed images.
          keep_prob: probability of keeping neurons intact.
          
        Returns:
          A downsampled segmentation mask. 
        """

        v_idx = 0 # Index variable.

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            w = self.variables[v_idx * 2]
            b = self.variables[v_idx * 2 + 1]
            conv = tf.nn.conv2d(input_batch, w, strides=[1, 1, 1, 1], padding='SAME')
            self.conv1_1 = tf.nn.relu(tf.nn.bias_add(conv, b))
            v_idx += 1

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            w = self.variables[v_idx * 2]
            b = self.variables[v_idx * 2 + 1]
            conv = tf.nn.conv2d(self.conv1_1, w, strides=[1, 1, 1, 1], padding='SAME')
            self.conv1_2 = tf.nn.relu(tf.nn.bias_add(conv, b))
            v_idx += 1

        # pool1
        # self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 3, 3, 1], strides=[1,2,2,1], padding='SAME' )
        self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME' )

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            w = self.variables[v_idx * 2]
            b = self.variables[v_idx * 2 + 1]
            conv = tf.nn.conv2d(self.pool1, w, strides=[1, 1, 1, 1], padding='SAME')
            self.conv2_1 = tf.nn.relu(tf.nn.bias_add(conv, b))
            v_idx += 1

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            w = self.variables[v_idx * 2]
            b = self.variables[v_idx * 2 + 1]
            conv = tf.nn.conv2d(self.conv2_1, w, strides=[1, 1, 1, 1], padding='SAME')
            self.conv2_2 = tf.nn.relu(tf.nn.bias_add(conv, b))
            v_idx += 1

        # pool2
        # self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            w = self.variables[v_idx * 2]
            b = self.variables[v_idx * 2 + 1]
            conv = tf.nn.conv2d(self.pool2, w, strides=[1, 1, 1, 1], padding='SAME')
            self.conv3_1 = tf.nn.relu(tf.nn.bias_add(conv, b))
            v_idx += 1

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            w = self.variables[v_idx * 2]
            b = self.variables[v_idx * 2 + 1]
            conv = tf.nn.conv2d(self.conv3_1, w, strides=[1, 1, 1, 1], padding='SAME')
            self.conv3_2 = tf.nn.relu(tf.nn.bias_add(conv, b))
            v_idx += 1

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            w = self.variables[v_idx * 2]
            b = self.variables[v_idx * 2 + 1]
            conv = tf.nn.conv2d(self.conv3_2, w, strides=[1, 1, 1, 1], padding='SAME')
            self.conv3_3 = tf.nn.relu(tf.nn.bias_add(conv, b))
            v_idx += 1

        # pool3
        # self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            w = self.variables[v_idx * 2]
            b = self.variables[v_idx * 2 + 1]
            conv = tf.nn.conv2d(self.pool3, w, strides=[1, 1, 1, 1], padding='SAME')
            self.conv4_1 = tf.nn.relu(tf.nn.bias_add(conv, b))
            v_idx += 1

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            w = self.variables[v_idx * 2]
            b = self.variables[v_idx * 2 + 1]
            conv = tf.nn.conv2d(self.conv4_1, w, strides=[1, 1, 1, 1], padding='SAME')
            self.conv4_2 = tf.nn.relu(tf.nn.bias_add(conv, b))
            v_idx += 1

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            w = self.variables[v_idx * 2]
            b = self.variables[v_idx * 2 + 1]
            conv = tf.nn.conv2d(self.conv4_2, w,strides=[1, 1, 1, 1], padding='SAME')
            self.conv4_3 = tf.nn.relu(tf.nn.bias_add(conv, b))
            v_idx += 1

        # pool4
        # self.pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        self.pool4 = self.conv4_3

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            w = self.variables[v_idx * 2]
            b = self.variables[v_idx * 2 + 1]
            conv = tf.nn.atrous_conv2d(self.pool4, w, rate=2, padding='SAME')
            self.conv5_1 = tf.nn.relu(tf.nn.bias_add(conv, b))
            v_idx += 1

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            w = self.variables[v_idx * 2]
            b = self.variables[v_idx * 2 + 1]
            conv = tf.nn.atrous_conv2d(self.conv5_1, w, rate=2, padding='SAME')
            self.conv5_2 = tf.nn.relu(tf.nn.bias_add(conv, b))
            v_idx += 1

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            w = self.variables[v_idx * 2]
            b = self.variables[v_idx * 2 + 1]
            conv = tf.nn.atrous_conv2d(self.conv5_2, w, rate=2, padding='SAME')
            self.conv5_3 = tf.nn.relu(tf.nn.bias_add(conv, b))
            v_idx += 1

        # pool5
        # self.pool5 = tf.nn.max_pool(self.conv5_3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        # pool6
        # self.pool6 = tf.nn.avg_pool(self.pool5, ksize =[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        self.pool6 = self.conv5_3

        # fc6 -> atrous_convolution
        with tf.name_scope('fc6') as scope:
            #w = self.variables[v_idx * 2]
            #b = self.variables[v_idx * 2 + 1]
            w = create_variable('fc6/w', [3, 3, 512, 1024])
            b = create_bias_variable('fc6/b', [1024])
            conv = tf.nn.atrous_conv2d(self.pool6, w, rate=4, padding='SAME')
            self.fc6 = tf.nn.relu(tf.nn.bias_add(conv, b))
            v_idx += 1

        # fc7 -> convolution
        with tf.name_scope('fc7') as scope:
            w = create_variable('fc7/w', [1, 1, 1024, 128])
            b = create_bias_variable('fc7/b', [128])
            conv = tf.nn.conv2d(self.fc6, w, strides=[1, 1, 1, 1],padding='SAME')
            self.fc7 = tf.nn.relu(tf.nn.bias_add(conv, b))

        # scnn layer
        self.scnn = create_scnn(self.fc7)
        #self.scnn = self.fc7

        # dropout
        # self.dropout = tf.nn.dropout(self.scnn, keep_prob=keep_prob)


        # fc8: No Relu
        with tf.name_scope('output1') as scope:
            w = create_variable('fc8/w', [1, 1, 128, 5])
            b = create_bias_variable('fc8/b', [5])
            conv = tf.nn.conv2d(self.scnn, w, strides=[1, 1, 1, 1], padding='SAME')
            self.fc8 = tf.nn.bias_add(conv, b)

        #return self.fc8


        # bilinear upsmapling


        # output2
        with tf.name_scope('output2') as scope:
            self.softmax = tf.nn.softmax(self.fc8, dim=3, name='spatial_softmax')
            self.avg_pool = tf.nn.avg_pool(self.softmax, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='avg_pool')
            # fc
            #shape = tf.shape(self.avg_pool)
            #shape = self.avg_pool.get_shape()
            #pool_flat = tf.reshape(self.avg_pool, [-1, shape[1] * shape[2] * shape[3] ])
            pool_flat = tf.reshape(self.avg_pool, [-1, 32 * 40 * 5])
            self.fc128 = tf.layers.dense(inputs=pool_flat, units=128, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer, name='fc128')
            # no sigmoid activation, sigmoid is included in sigmoid_cross_entropy
            self.fc4 = tf.layers.dense(self.fc128, 4, activation=None, kernel_initializer=tf.truncated_normal_initializer, name='fc4')


        return self.fc8, self.fc4

    
    def prepare_label(self, input_batch, new_size):
        """Resize masks and perform one-hot encoding.

        Args:
          input_batch: input tensor of shape [batch_size H W 1].
          new_size: a tensor with new height and width.

        Returns:
          Outputs a tensor of shape [batch_size h w 21]
          with last dimension comprised of 0's and 1's only.
        """
        with tf.name_scope('label_encode'):
            input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # As labels are integer numbers, need to use nearest neighbour interpolation.
            input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # Reducing the channel dimension.
            input_batch = tf.one_hot(tf.cast(input_batch, tf.int32), depth=5)
        return input_batch
      
    def preds(self, input_batch):
        """Create the network and run inference on the input batch.
        
        Args:
          input_batch: batch of pre-processed images.
          
        Returns:
          Argmax over the predictions of the network of the same shape as the input.
        """
        outpu1, output2 = self._create_network(tf.cast(input_batch, tf.float32), keep_prob=tf.constant(1.0))
        outpu1 = tf.image.resize_bilinear(outpu1, tf.shape(input_batch)[1:3])
        outpu1 = tf.argmax(outpu1, dimension=3)
        outpu1 = tf.expand_dims(outpu1, dim=3) # Create 4D-tensor.
        return tf.cast(outpu1, tf.uint8), tf.sigmoid(output2)
        
    
    def loss(self, img_batch, label_batch1, label_batch2):
        """Create the network, run inference on the input batch and compute loss.
        
        Args:
          input_batch: batch of pre-processed images.
          
        Returns:
          Pixel-wise softmax loss.
        """
        output1, output2 = self._create_network(tf.cast(img_batch, tf.float32), keep_prob=tf.constant(DROPOUT))
        #print 'output1 shape:', output1.shape
        #print 'output2 shape:', output2.shape
        prediction1 = tf.reshape(output1, [-1, n_classes])

        
        # Need to resize labels and convert using one-hot encoding.
        # label_batch = self.prepare_label(label_batch1, tf.pack(outpu1.get_shape()[1:3]))
        label_batch = self.prepare_label(label_batch1, output1.shape[1:3])
        gt1 = tf.reshape(label_batch, [-1, n_classes])
        
        # Pixel-wise softmax loss.
        assert gt1.get_shape() == prediction1.get_shape()
        print 'groundTruth shape: ', gt1.get_shape()
        print 'prediction1 shape: ', prediction1.get_shape()
        #loss1 = tf.nn.softmax_cross_entropy_with_logits(labels=gt1, logits=prediction1)
        class_weights = tf.constant([[0.4, 1.0, 1.0, 1.0, 1.0]])
        weights_loss = tf.reduce_sum(tf.multiply(gt1, class_weights), 1)
        loss1 = tf.losses.softmax_cross_entropy(onehot_labels=gt1, logits=prediction1, weights=weights_loss)
        reduced_loss1 = tf.reduce_mean(loss1)


        # sigmoid loss
        prediction2 = tf.reshape(tf.cast(output2, tf.float32), [-1, 4])
        gt2 = tf.reshape(tf.cast(label_batch2, tf.float32), [-1, 4])
        loss2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt2, logits=prediction2)
        reduced_loss2 = tf.reduce_mean(loss2)

        loss = reduced_loss1 + 0.1 * reduced_loss2

        # debug
        #pdb.set_trace()
        #return reduced_loss1, reduced_loss2, loss, prediction1, gt1, prediction2, gt2
        

        return reduced_loss1, reduced_loss2, loss

