import tensorflow as tf
import mnist_inference
import os

# load function which generate PROJECTOR log
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

# super parameter
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99

# other parameters
LOG_DIR = '/home/harvey/tftest'
SPRITE_FILE = 'mnist_sprite.jpg'
META_FILE = 'mnist_meta.tsv'
TENSOR_NAME = 'FINAL_LOGITS'

# train
def train(mnist):
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULATION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # moving average
    with tf.name_scope('moving_average'):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # loss function
    with tf.name_scope('loss_function'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # train step
    with tf.name_scope('train_step'):
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY, staircase=True)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name='train')

    # train model
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:xs, y_:ys})
            if i%1000 == 0:
                print ("After %d training step, loss on training batch is %g. " % (i, loss_value))
        final_result = sess.run(y, feed_dict={x:mnist.test.images})
    return final_result

# generate log for visualization
def visualization(final_result):
    # define a variable to store final_result
    y = tf.Variable(final_result, name=TENSOR_NAME)
    summary_writer = tf.summary.FileWriter(LOG_DIR)
    # config class help to generate log files
    config = projector.ProjectorConfig()
    # add a visualize embbedding result
    embedding = config.embeddings.add()
    # assign the embedding variable name
    embedding.tensor_name = y.name
    # assign the result meta info, such as label (optional)
    embedding.metadata_path = META_FILE
    # assign sprite img (optional)
    embedding.sprite.image_path = SPRITE_FILE
    # single sptite img size
    embedding.sprite.single_image_dim.extend([28,28])

    # write the info into log for PROJECTOR
    projector.visualize_embeddings(summary_writer, config)

    # new sess
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(LOG_DIR, "model"), TRAINING_STEPS)

    summary_writer.close()

# main function
def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    final_result = train(mnist)
    visualization(final_result)

if __name__ == '__main__':
    main()


