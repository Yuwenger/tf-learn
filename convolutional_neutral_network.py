from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/mnist/", one_hot=True)

import tensorflow as tf

# Training Parameters
learning_rate = 0.001
num_steps = 200
batch_size = 128
display_step = 100

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
X = tf.placeholder(tf.float32,[None,784])
Y = tf.placeholder(tf.int32,[None,num_classes])
dropout = tf.placeholder(tf.float32)
def conv_net(x,n_classes,keep_prob,reuse,is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.layers.dense(fc1, 1024)
        fc1 = tf.layers.dropout(fc1, rate=keep_prob, training=is_training)
        out = tf.layers.dense(fc1, n_classes)
    return out

logits_train = conv_net(X,num_classes,dropout,False,True)
logits_test = conv_net(X,num_classes,dropout,True,False)
pred_class = tf.argmax(logits_test,axis=1)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_train,labels=Y))
loss_test = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_test,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op,global_step=tf.train.get_global_step())
acc_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y,axis=1),tf.argmax(logits_test,axis=1)),tf.float32))

init = tf.global_variables_initializer()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)
    for step in range(1, num_steps + 1):

        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, dropout: 0.8})
        if step % display_step == 0:
            # print step
            loss, acc = sess.run([loss_test, acc_op], feed_dict={X: batch_x, Y: batch_y, dropout: 1.0})
            print 'step {},loss = {:.4f},acc = {:.3f}'.format(step, loss, acc)
    print 'finish.'
    print 'testing .....'
    print 'final acc: {:.3f}'.format(sess.run(acc_op, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))



