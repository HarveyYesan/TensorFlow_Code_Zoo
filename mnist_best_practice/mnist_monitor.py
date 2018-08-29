import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

SUMMARY_DIR = "/home/harvey/tftest"
BATCH_SIZE = 100
TRAIN_STEPS = 3000

# generate variable monitoring info
def variable_summary(var, name):
    with tf.name_scope('summaries'):
        # record values distribution in tensor
        tf.summary.histogram(name, var)
        # calculate mean of variable
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        # calculate stddev
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)

# define a fully connected layer
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            variable_summary(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0, shape=[output_dim]))
            variable_summary(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            # record the distribution over the output before activate function
            tf.summary.histogram(layer_name + '/preactivate', preactivate)
        activations = act(preactivate, name='activation')
        # record the distribution over the output before activate function
        tf.summary.histogram(layer_name + '/activations', activations)
        return activations

def main(_):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
    # recovery the input vector to pixel matrix, write current images info to log
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)
    hidden1 = nn_layer(x, 784, 500, 'layer1')
    y = nn_layer(hidden1, 500, 10, 'layer2', act=tf.identity)

    # calculate cross_entropy and record info in log
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)) 
        tf.summary.scalar('cross_entropy', cross_entropy)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # evaluate and record info in log
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # merge all log info
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        # init writer
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        tf.global_variables_initializer().run()
        for i in range(TRAIN_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # train and generate log info
            summary, _ = sess.run([merged, train_step], feed_dict={x:xs, y_:ys})
            # write the log info into file, for TensorBoard reading
            writer.add_summary(summary, i)
            print ("train step %d " % i)
    writer.close()

if __name__ == '__main__':
    tf.app.run()




