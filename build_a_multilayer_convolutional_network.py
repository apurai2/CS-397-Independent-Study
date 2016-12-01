import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Common usage for TensorFlow is to first create a graph and then
# launch it in a session.
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# Initialize weights with a small amount of noise for symmetry
# breaking and to prevent 0 gradients.
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Convolution and pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# Convolutional will compute 32 features for each 5x5 patch. The
# first two dimensions are the patch size, the next is the number
# of input channels, and the last is the number of output channels.
# We will also have a bias vector with a component for each
# output channel
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# Reshape x to a 4d tensor, with second and third dimensions
# corresponding to image width and height, and the final dimension
# corresponding to the number of color channels.
x_image = tf.reshape(x, [-1, 28, 28, 1])

# First convolutional layer
# Convolve x_image with the weight tensor, add bias, and apply
# ReLU function, and finally max pool.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second convolutional layer
# Has 64 features for each 5x5 patch
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely connected layer
# Add a fully-connected layer with 1024 neurons to allow processing
# on the entire image. Reshape tensor from pooling layer into a
# batch of vectors, multiply by a weight matrix, add a bias, and
# apply ReLU.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Apply dropout before readout layer to reduce overfitting. Create
# a plceholder for the probability that a neuron's output is kept
# during dropout, which allows us to turn dropout on during training
# and off during testing.
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Add a softmax readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Train and evaluate the model
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(1000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0
        })
        print("Step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={
        x: batch[0], y_: batch[1], keep_prob: 0.5
    })

print("Test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0
}))

# For this small convolutional network, performace is nearly
# identical with and without dropout. Dropout is effective at
# reducing overfitting, but it is most useful when training very
# large neural networks.
