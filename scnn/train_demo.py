import tensorflow as tf
from scnn_model import SCNNModel

image = tf.ones(shape=[1, 512, 640, 3], dtype=tf.float32)
label_batch1 = tf.ones([1, 512, 640, 1], tf.int32)
label_batch2 = tf.ones([1, 4], tf.int32)

path_pretrained_params = './pretrained_params/pretrained_params.ckpt'
scnn = SCNNModel(path_pretrained_params)
loss = scnn.loss(image, label_batch1, label_batch2)
trainable = tf.trainable_variables()
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list=trainable)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(10):
    loss_value , _ = sess.run([loss, optimizer])
    print 'loss of ', i, 'iteration: ', loss_value
