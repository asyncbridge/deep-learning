import tensorflow as tf


def pre_process_image(image):
    # Just resize CIFAR-100 image to ImageNet image size to fit to the CNN network architecture in the paper.
    # but output classes are same as CIFAR-100.
    image = tf.image.resize_images(image, [256, 256])

    # #1, Random Crop and Horizontal Reflection
    image = tf.random_crop(image, [224, 224, 3])
    image = tf.image.random_flip_left_right(image)

    # #2, Altering RGB Intensities
    image = tf.image.random_brightness(image, max_delta=0.4)
    image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
    image = tf.image.random_saturation(image, lower=0.6, upper=1.4)

    return image


def augment_image(input_tensor):
    output_tensor = tf.map_fn(pre_process_image, input_tensor)
    return output_tensor

class AlexNetModel(object):
    def __init__(self, weight_decay=0.0005):

        tf.set_random_seed(1234)

        self.weight_decay = weight_decay

        # Input Layer
        with tf.name_scope("input") as scope:

            self.dropout = tf.placeholder(tf.float32, name="dropout")

            # Input CIFAR-100 image size, 32x32x3
            # Color Channel 3(RGB)
            self.input_x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="input_x")

            self.image_summary_original = tf.summary.image("image_original", self.input_x)

            # Augmentation
            # #1, Random Crop and Horizontal Reflection
            # #2, Altering RGB Intensities
            augmented_input_x = augment_image(self.input_x)

            self.image_summary_augmented = tf.summary.image("image_augmented", augmented_input_x)

            # CIFAR-100 Label (0~99), one-hot vector
            self.input_y= tf.placeholder(tf.int32, [None, 1], name="input_y")

            input_y_one_hot = tf.one_hot(self.input_y, 100)
            input_y_one_hot = tf.reshape(input_y_one_hot, [-1, 100])

        with tf.name_scope("conv1") as scope:
                conv1 = self.conv(augmented_input_x, 11, 11, 3, 96, 4, 4, "VALID", "conv1")
                conv1 = self.lrn(conv1, 2.0, 10e-4, 0.75, 1.0)
                conv1 = self.max_pool(conv1, 3, 3, 2, 2, "VALID", "max_pool1")

        with tf.name_scope("conv2") as scope:
                conv2 = self.conv(conv1, 5, 5, 96, 256, 1, 1, "SAME", "conv2", False)
                conv2 = self.lrn(conv2, 2.0, 10e-4, 0.75, 1.0)
                conv2 = self.max_pool(conv2, 3, 3, 2, 2, "VALID", "max_pool2")

        with tf.name_scope("conv3") as scope:
                conv3 = self.conv(conv2, 3, 3, 256, 384, 1, 1, "SAME", "conv3")

        with tf.name_scope("conv4") as scope:
                conv4 = self.conv(conv3, 3, 3, 384, 384, 1, 1, "SAME", "conv4", False)

        with tf.name_scope("conv5") as scope:
                conv5 = self.conv(conv4, 3, 3, 384, 256, 1, 1, "SAME", "conv5", False)
                conv5 = self.max_pool(conv5, 3, 3, 2, 2, "VALID", "max_pool5")

        with tf.name_scope("fc6") as scope:
            flattened_size = 5 * 5 * 256  # 6 * 6 * 256
            flattened = tf.reshape(conv5, [-1, flattened_size])

            fc6 = self.fc(flattened, flattened_size, 4096, "fc6", True, False)
            fc6 = tf.nn.dropout(fc6, keep_prob=self.dropout)

        with tf.name_scope("fc7") as scope:
            fc7 = self.fc(fc6, 4096, 4096, "fc7", True, False)
            fc7 = tf.nn.dropout(fc7, keep_prob=self.dropout)

        with tf.name_scope("fc8") as scope:
            self.hypothesis = self.fc(fc7, 4096, 100, "fc8", False, False)

        # Loss Function
        with tf.name_scope("loss") as scope:
            L2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.hypothesis, labels=input_y_one_hot), name="loss") + tf.reduce_sum(L2_loss)

        # Accuracy
        with tf.name_scope("accuracy") as scope:
            self.predictions = tf.argmax(self.hypothesis, 1, name="predictions")
            correct_predictions = tf.equal(self.predictions, tf.argmax(input_y_one_hot, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def conv(self, input_tensor, filter_height, filter_width, in_num_of_filter, out_num_of_filter, stride_y, stride_x, padding, name, is_bias_zero=True):
        with tf.variable_scope(name):
            w = tf.get_variable(
                    "weight",
                    initializer=tf.random_normal([filter_height, filter_width, in_num_of_filter, out_num_of_filter], stddev=0.01),
                    regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))

            bias_initializer = tf.zeros([out_num_of_filter])

            if is_bias_zero == False:
                bias_initializer = tf.ones([out_num_of_filter])

            b = tf.get_variable(
                    "bias",
                    initializer=bias_initializer)

        conv = tf.nn.conv2d(input_tensor, w, strides=[1, stride_y, stride_x, 1], padding=padding, name=name)
        conv = tf.nn.bias_add(conv, b)
        conv = tf.nn.relu(conv)

        return conv

    def lrn(self, input_tensor, depth_radius, alpha, beta, bias=1.0):
        return tf.nn.local_response_normalization(input_tensor, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta)

    def max_pool(self, input_tensor, ksize_y, ksize_x, stride_y, stride_x, padding, name):
        return tf.nn.max_pool(input_tensor, ksize=[1, ksize_y, ksize_x, 1], strides=[1, stride_y, stride_x, 1], padding=padding, name=name)

    def fc(self, input_tensor, in_fc, out_fc, name, is_relu=True, is_bias_zero=True):
        with tf.variable_scope(name):
            w = tf.get_variable(
                    "weight",
                    initializer=tf.random_normal([in_fc, out_fc], stddev=0.01),
                    regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))

            bias_initializer = tf.zeros([out_fc])

            if is_bias_zero == False:
                bias_initializer = tf.ones([out_fc])

            b = tf.get_variable(
                "bias",
                initializer=bias_initializer)

        fc = tf.nn.xw_plus_b(input_tensor, w, b)

        if is_relu == True:
            fc = tf.nn.relu(fc)

        return fc