import tensorflow as tf
from six.moves import cPickle
import os

# Loading net skeleton with parameters name and shapes.
with open("../net_skeleton.ckpt", "rb") as f:
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
n_classes = 21
# All convolutional and pooling operations are applied using kernels of size 3x3; 
# padding is added so that the output of the same size as the input.
ks = 3

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

class DeepLabLFOVModel(object):
    """DeepLab-LargeFOV model with atrous convolution and bilinear upsampling.
    
    This class implements a multi-layer convolutional neural network for semantic image segmentation task.
    This is the same as the model described in this paper: https://arxiv.org/abs/1412.7062 - please look
    there for details.
    """
    
    def __init__(self, weights_path=None):
        """Create the model.
        
        Args:
          weights_path: the path to the cpkt file with dictionary of weights from .caffemodel.
        """
        self.variables = self._create_variables(weights_path)
        input = tf.placeholder(tf.float32, [4, 512, 640, 3], name='input')
        self._create_network(input, 0.5)

        
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
                for name, shape in net_skeleton:
                    var.append(tf.Variable(weights[name],
                                           name=name))
                del weights
        else:
            # Initialise all weights randomly with the Xavier scheme,
            # and 
            # all biases to 0's.
            for name, shape in net_skeleton:
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
        current = input_batch
        
        v_idx = 0 # Index variable.
        
        # Last block is the classification layer.
        for b_idx in xrange(len(dilations) - 1):
            for l_idx, dilation in enumerate(dilations[b_idx]):
                w = self.variables[v_idx * 2]
                b = self.variables[v_idx * 2 + 1]
                if dilation == 1:
                    conv = tf.nn.conv2d(current, w, strides=[1, 1, 1, 1], padding='SAME')
                else:
                    conv = tf.nn.atrous_conv2d(current, w, dilation, padding='SAME')
                current = tf.nn.relu(tf.nn.bias_add(conv, b))
                v_idx += 1
            # Optional pooling and dropout after each block.
            if b_idx < 3:
                current = tf.nn.max_pool(current, 
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 2, 2, 1],
                                         padding='SAME')
            elif b_idx == 3:
                current = tf.nn.max_pool(current, 
                             ksize=[1, ks, ks, 1],
                             strides=[1, 1, 1, 1],
                             padding='SAME')
            elif b_idx == 4:
                current = tf.nn.max_pool(current, 
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
                current = tf.nn.avg_pool(current, 
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
            elif b_idx <= 6:
                current = tf.nn.dropout(current, keep_prob=keep_prob)
        
        # Classification layer; no ReLU.
        w = self.variables[v_idx * 2]
        b = self.variables[v_idx * 2 + 1]
        conv = tf.nn.conv2d(current, w, strides=[1, 1, 1, 1], padding='SAME')
        current = tf.nn.bias_add(conv, b)

        return current


    def get_conv_params(self):
        # conv1_1/b:0, conv1_1/w:0
        return {var.name[:-2]: var for var in tf.global_variables() if 'conv' in var.name}


    def save_convs_params(self, sess, save_dir):
        #conv_weights = dict()
        conv_weights = sess.run( self.get_conv_params() )

        # for name, shape in net_skeleton:
        #     if 'conv' == name[0:4]:
        #         conv_weights[name] =

        #print(type(conv_weights))
        print(conv_weights['conv1_1/b'])
        print(conv_weights.keys())

        with open(os.path.join(save_dir, 'pretrained_params.ckpt'), 'wb') as f:
            cPickle.dump(conv_weights, f, protocol=cPickle.HIGHEST_PROTOCOL)

