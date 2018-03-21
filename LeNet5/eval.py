import tensorflow as tf
import os
import helper
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import csv

print("Tensorflow Ver.={}".format(tf.__version__))

tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Evaluation
print("\nEvaluating...\n")

# ex) "./runs/1521530071/checkpoints"
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)

    sess = tf.Session(config=session_conf)

    with sess.as_default():
        with tf.device('/cpu:0'):
            # Label data is no one-hot encoding.
            mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

            # Test Data: 10,000
            X_raw, y_test = mnist.test.images, mnist.test.labels

            # Reshape input data to 28x28 size to 32x32 size with zero-padding
            X_test = X_raw.reshape(-1, 28, 28, 1)
            X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("Layer_Input/InputX").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("Layer_Output/Predictions").outputs[0]

            # Generate batches for one epoch
            batches = helper.Generate_Batches(list(X_test), 1, 50, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

            # Accuracy for test data
            correct_predictions = float(sum(all_predictions == y_test))
            print("Total number of test examples: {}".format(len(y_test)))
            print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))