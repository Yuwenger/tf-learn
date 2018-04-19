import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.01
num_steps = 5000
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
X = tf.placeholder('float',[None,num_input])
Y = tf.placeholder('float',[None,num_classes])

def neutral_net(x):
    layer_1 = tf.layers.dense(x,n_hidden_1)
    layer_2 = tf.layers.dense(layer_1,n_hidden_2)
    out_layer = tf.layers.dense(layer_2,num_classes)
    return out_layer

logits = neutral_net(X)
prediction = tf.nn.softmax(logits)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
cor = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
acc = tf.reduce_mean(tf.cast(cor,tf.float32))
init = tf.global_variables_initializer()
with tf.Session()  as sess:
    sess.run(init)
    for step in range(1,num_steps+1):
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_op,feed_dict={X:batch_x,Y:batch_y})
        if step% display_step == 0:
            loss,acc_ = sess.run([loss_op,acc],feed_dict={X:batch_x,Y:batch_y})
            print 'step {}, loss = {}, acc = {}'.format(step,loss,acc_)
    print 'op finish!'
    print 'test model...'
    acc_ = sess.run(acc,feed_dict = {X:mnist.test.images,Y:mnist.test.labels})
    print 'final acc: {}'.format(acc_)



