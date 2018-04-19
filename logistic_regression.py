from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("./data/mnist/", one_hot=True)
print tf.__version__
lr = 0.01
epo = 25
batch_size = 100
dis_step = 1
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
pred = tf.nn.softmax(tf.matmul(x,w)+b)
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices = 1))
op= tf.train.GradientDescentOptimizer(lr).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for ep in range(epo):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            _,c = sess.run([op,cost],feed_dict = {x:batch_x,y:batch_y})
            avg_cost+= c/total_batch
        if ep % dis_step == 0:
             print 'epoch {}: cost = {}'.format(ep,avg_cost)
    print 'op finish!'
    cor = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    acc = tf.reduce_mean(tf.cast(cor,tf.float32))
    print 'acc: {}'.format(acc.eval({x:mnist.test.images,y:mnist.test.labels}))
