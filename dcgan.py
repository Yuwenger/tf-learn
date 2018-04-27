from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/mnist/", one_hot=True)

import tensorflow as tf
num_steps = 2000
batch_size = 32

image_dim = 784
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 200

def generator(x,reuse= False):
    with tf.variable_scope('Generator',reuse=reuse):
        x = tf.layers.dense(x,units=6*6*128)
        x = tf.nn.tanh(x)
        x = tf.reshape(x,shape=[-1,6,6,128])
        x = tf.layers.conv2d_transpose(x,64,4,strides=2)
        x = tf.layers.conv2d_transpose(x,1,2,strides=2)
        x = tf.nn.sigmoid(x)
        return x
def discriminator(x,reuse=False):
    with tf.variable_scope('Discriminator',reuse=reuse):
        x = tf.layers.conv2d(x,64,5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x,2,2)
        x= tf.layers.conv2d(x,128,5)
        x = tf.nn.ranh(x)
        x = tf.layers.average_pooling2d(x,2,2)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x,1024)
        x = tf.nn.tanh(x)
        x = tf.layers.dense(x,2)
        return x

noise_input = tf.placeholder(tf.float32,shape=[None,noise_dim])
real_image_input = tf.placeholder(tf.float32,shape=[None,28,28,1])
gen_sample = generator(noise_input)
disc_real = discriminator(real_image_input)
disc_fake = discriminator(gen_sample,reuse=True)

