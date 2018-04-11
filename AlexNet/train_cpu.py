import tensorflow as tf
import os
import time
import datetime
import helper
from alex_model_cpu import AlexNetModel
from tensorflow.python.keras._impl.keras.datasets.cifar100 import load_data
import numpy as np
import matplotlib.pyplot as plt

tf.flags.DEFINE_integer("num_of_checkpoints", 5, "Number of Checkpoints")
tf.flags.DEFINE_float("cross_val_percent", 0.1, "The ratio of cross-validation set in the training set")
tf.flags.DEFINE_integer("epochs", 10, "Epochs")
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size")
tf.flags.DEFINE_float("learning_rate", 1e-2, "Learning Rate")
tf.flags.DEFINE_integer("cross_validation_step_once", 100, "Cross-Validation after this step")
tf.flags.DEFINE_integer("checkpoint_save_once", 100, "Save checkpoint after this step")

FLAGS = tf.flags.FLAGS

print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().items()) :
    print("{}={}".format(attr.upper(), value))

print("\nTraining...\n")

def main(argv=None):
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # Load CIFAR-100 dataset
        (X_train, y_train), (X_test, y_test) = load_data()

        # Separate train, cross validation, test data
        # Training Data: 45,000
        # Cross-Validation Data: 5,000
        # Test Data: 10,000

        # Randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(X_train)))
        x_shuffled = X_train[shuffle_indices]
        y_shuffled = y_train[shuffle_indices]

        # Split train/cross-validation set
        dev_sample_index = -1 * int(FLAGS.cross_val_percent * float(len(X_train)))
        X_train, X_validation = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_validation = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        print("Split -- Train/Cross-Validation: {:d}/{:d}\n".format(len(y_train), len(y_validation)))

        # Check Shape
        print("Dataset shape: ")
        print("X_train.shape={}", X_train.shape)
        print("y_train.shape={}", y_train.shape)
        print("X_validation.shape={}", X_validation.shape)
        print("y_validation.shape={}", y_validation.shape)
        print("X_test.shape={}", X_test.shape)
        print("y_test.shape={}", y_test.shape)

        # Create a AlexNet model
        cnn = AlexNetModel()

        session_conf = tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False)

        sess = tf.Session(config=session_conf)

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
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged, cnn.image_summary_original, cnn.image_summary_augmented])
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

        for batch in batches:
            x_batch, y_batch = zip(*batch)

            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout: 0.5
            }

            _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.cross_validation_step_once == 0:
                print("\nCross-Validation:")

                feed_dict = {
                    cnn.input_x: X_validation,
                    cnn.input_y: y_validation,
                    cnn.dropout: 1.0
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

if __name__ == '__main__':
    tf.app.run()