import tensorflow as tf
from model import SCNNModel
from six.moves import cPickle


if __name__ == '__main__':

    input = tf.placeholder(tf.float32, [4, 512, 640, 3], name='input')
    scnn = SCNNModel()
    # with open('./pretrained_model/init_model.ckpt', "rb") as f:
    #     weights = cPickle.load(f)
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    writer = tf.summary.FileWriter('./', sess.graph)