import tensorflow as tf
import os
import time
import datetime
import helper
from lenet5_model import LeNet5Model
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

tf.flags.DEFINE_string("num_of_checkpoints", 5, "Number of Checkpoints")
tf.flags.DEFINE_string("epochs", 10, "Epochs")
tf.flags.DEFINE_string("batch_size", 128, "Batch Size")
tf.flags.DEFINE_string("learning_rate", 1e-3, "Learning Rate")
tf.flags.DEFINE_string("cross_validation_step_once", 100, "Cross-Validation after this step")
tf.flags.DEFINE_string("checkpoint_save_once", 100, "Save checkpoint after this step")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

print("\nTraining...\n")

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)

    sess = tf.Session(config=session_conf)

    with sess.as_default():
        with tf.device('/cpu:0'):
            # Label data is one-hot encoding and reshape is False. It means that the image size is 28x28x1 not 784.
            mnist = input_data.read_data_sets("MNIST_data/", reshape=False, one_hot=True)

            # Separate train, cross validation, test data
            # Training Data: 55,000
            # Cross-Validation Data: 5,000
            # Test Data: 10,000
            X_train, y_train = mnist.train.images, mnist.train.labels
            X_validation, y_validation = mnist.validation.images, mnist.validation.labels
            X_test, y_test = mnist.test.images, mnist.test.labels

            # Check Shape
            print("X_train.shape={}".format(X_train.shape))
            print("y_train.shape={}".format(y_train.shape))

            # Reshape input data to 28x28 size to 32x32 size with zero-padding
            X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            X_validation = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

            # Create a LeNet-5 model
            cnn = LeNet5Model()

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Cross-Validation summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "cross-validation")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_of_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Train LeNet-5 model
            batches = helper.Generate_Batches(list(zip(X_train, y_train)), FLAGS.epochs, FLAGS.batch_size)

            print(batches)

            for batch in batches:
                x_batch, y_batch = zip(*batch)

                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch
                }

                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.cross_validation_step_once == 0:
                    print("\nCross-Validation:")

                    feed_dict = {
                        cnn.input_x: X_validation,
                        cnn.input_y: y_validation
                    }

                    step, summaries, loss, accuracy = sess.run(
                        [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    dev_summary_writer.add_summary(summaries, step)
                    print("")

                if current_step % FLAGS.checkpoint_save_once == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))