import tensorflow as tf

class AlexNetModel(object):
    def __init__(self, learning_rate=0.01):
        # Input Layer
        with tf.name_scope("Layer_Input") as scope:
            # Input CIFAR-100 image size, 32x32x3
            # Color Channel 3(RGB)
            self.input_x = tf.placeholder(tf.float32, [None, 32, 32, 1], name="InputX")

            # MNIST Label (0~9), one-hot vector
            self.input_y = tf.placeholder(tf.float32, [None, 10], name="InputY")

        # C1 Layer
        with tf.name_scope("Layer_C1") as scope:
            # 6@5x5 filter
            W1 = tf.Variable(tf.random_normal([5, 5, 1, 6], stddev=0.1), name="W1")
            B1 = tf.Variable(tf.random_normal([6], stddev=0.1), name="B1")

            # No Zero Padding("VALID"), stride=1
            C1 = tf.nn.conv2d(self.input_x, W1, strides=[1, 1, 1, 1], padding="VALID", name="C1")
            C1 = tf.nn.bias_add(C1, B1)
            C1 = tf.nn.relu(C1)

        # S2 Layer
        with tf.name_scope("Layer_S2") as scope:
            # 6@2x2 filter, average pooling, No Zero Padding("VALID"), stride=2
            #B2 = tf.Variable(tf.random_normal([6], stddev=0.1), name="B2")
            #S2 = tf.nn.avg_pool(C1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="S2")
            #S2 = tf.nn.bias_add(S2, B2)
            #S2 = tf.nn.sigmoid(S2)
            S2 = tf.nn.max_pool(C1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="S2")

        # C3 Layer
        with tf.name_scope("Layer_C3") as scope:
            # 16@5x5 filter
            W3 = tf.Variable(tf.random_normal([5, 5, 6, 16], stddev=0.1), name="W3")
            B3 = tf.Variable(tf.random_normal([16], stddev=0.1), name="B3")

            # No Zero Padding("VALID"), stride=1
            C3 = tf.nn.conv2d(S2, W3, strides=[1, 1, 1, 1], padding="VALID", name="C3")
            C3 = tf.nn.bias_add(C3, B3)
            C3 = tf.nn.relu(C3)

        # S4 Layer
        with tf.name_scope("Layer_S4") as scope:
            # 16@2x2 filter, average pooling, No Zero Padding("VALID"), stride=2
            #B4 = tf.Variable(tf.random_normal([16], stddev=0.1), name="B4")
            #S4 = tf.nn.avg_pool(C3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="S4")
            #S4 = tf.nn.bias_add(S4, B4)
            #S4 = tf.nn.sigmoid(S4)
            S4 = tf.nn.max_pool(C3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="S4")

        # C5 Layer
        with tf.name_scope("Layer_C5") as scope:
            # flatten S4
            S4_flat = tf.reshape(S4, [-1, 5 * 5 * 16])
            W5 = tf.Variable(tf.random_normal([5 * 5 * 16, 120], stddev=0.1), name="W5")
            B5 = tf.Variable(tf.random_normal([120], stddev=0.1), name="B5")

            C5 = tf.nn.xw_plus_b(S4_flat, W5, B5, name="C5")
            C5 = tf.nn.relu(C5)

        # F6 Layer
        with tf.name_scope("Layer_F6") as scope:
            W6 = tf.Variable(tf.random_normal([1 * 1 * 120, 84], stddev=0.1), name="W6")
            B6 = tf.Variable(tf.random_normal([84], stddev=0.1), name="B6")

            F6 = tf.nn.xw_plus_b(C5, W6, B6, name="F6")
            F6 = tf.nn.relu(F6)

        # Output Layer
        with tf.name_scope("Layer_Output") as scope:
            W7 = tf.Variable(tf.random_normal([1 * 1 * 84, 10], stddev=0.1), name="W7")
            B7 = tf.Variable(tf.random_normal([10], stddev=0.1), name="B7")

            self.hypothesis = tf.nn.xw_plus_b(F6, W7, B7, name="Hypothesis")

            self.predictions = tf.argmax(self.hypothesis, 1, name="Predictions")

        # Loss Function
        with tf.name_scope("Loss") as scope:
            #self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.hypothesis, labels=self.input_y), name="Loss")
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.hypothesis, labels=self.input_y), name="Loss")

        # Accuracy
        with tf.name_scope("Accuracy") as scope:
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="Accuracy")