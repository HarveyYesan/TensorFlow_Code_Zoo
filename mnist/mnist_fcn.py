import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# constatnt
# input nodes
INPUT_NODE = 784
# output nodes
OUTPUT_NODE = 10
# hidden layer node
LAYER1_NODE = 500
# batch size
BATCH_SIZE = 100
# super parameters
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# forward inference network
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # no moving average class
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    # using moving average class
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")

    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # inference result, none moving average class
    y = inference(x, None, weights1, biases1, weights2, biases2);

    # global traning steps
    global_step = tf.Variable(0, trainable=False)
    # init moving average class
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # using moving average on trainable variables
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # inference result, using moving average class
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # loss
    # cross entropy
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # l2 regularization
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization

    # learning rate decay
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    # optimize loss function
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # update training parameters and moving average value
    # one solution: train_op = tf.group(train_step, variables_averages_op)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # evaluate forward result of moving average
    correct_predictions = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # set up a session
    with tf.Session() as sess:
        # init
        tf.global_variables_initializer().run()
        # iteration
        for i in range(TRAINING_STEPS):
            if i % 100 == 0:
                validate_acc = sess.run(accuracy, feed_dict={x:mnist.validation.images, y_:mnist.validation.labels})
                print("After %d training steps, validation accuracy using average model is %g " % (i, validate_acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x:xs, y_:ys})

        # test model
        test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels})
        print("After %d training steps, test accuary using average model is %g" % (TRAINING_STEPS, test_acc))

# main function
def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    train(mnist)

# tensorflow main process api
if __name__ == '__main__':
    tf.app.run()

