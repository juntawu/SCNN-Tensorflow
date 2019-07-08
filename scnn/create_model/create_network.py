import tensorflow as tf
from scnn_model import SCNNModel
from six.moves import cPickle


if __name__ == '__main__':

    path_pretrained_params = '../pretrained_params/pretrained_params.ckpt'
    scnn = SCNNModel(path_pretrained_params)
    # with open('./pretrained_model/init_model.ckpt', "rb") as f:
    #     weights = cPickle.load(f)
    image_batch = tf.ones([4, 512, 640, 3], tf.float32)
    label_batch1 = tf.ones([4, 512, 640, 1], tf.int32)
    label_batch2 = tf.ones([4, 4], tf.int32)
    loss = scnn.loss(image_batch, label_batch1, label_batch2)
    sess = tf.Session()

    #init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    sess.run(init)
    #print sess.run(loss)
    #graph = tf.get_default_graph()
    #print sess.run(graph.get_tensor_by_name(name='conv1_1/w:0') )

    writer = tf.summary.FileWriter('./graph/', sess.graph)