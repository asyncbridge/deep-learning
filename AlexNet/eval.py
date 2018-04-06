import tensorflow as tf
import os
import helper
from tensorflow.python.keras._impl.keras.datasets.cifar100 import load_data
import numpy as np

print("Tensorflow Ver.={}".format(tf.__version__))

tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

FLAGS = tf.flags.FLAGS

print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().items()) :
    print("{}={}".format(attr.upper(), value))

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
            # Load CIFAR-100 dataset
            (X_train, y_train), (X_test, y_test) = load_data()

            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input/input_x").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("accuracy/predictions").outputs[0]

            # Generate batches for one epoch
            batches = helper.Generate_Batches(list(X_test), 1, 128, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

            # Accuracy for test data
            correct_predictions = float(sum(all_predictions == y_test))
            print("Total number of test examples: {}".format(len(y_test)))
            print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))