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

def conv(input_tensor, filter_height, filter_width, in_num_of_filter, out_num_of_filter, stride_y, stride_x, padding, name):
    w = tf.Variable(tf.random_normal([filter_height, filter_width, in_num_of_filter, out_num_of_filter], mean=0.0, stddev=0.01))
    b = tf.Variable(tf.zeros([out_num_of_filter]))

    conv = tf.nn.conv2d(input_tensor, w, strides=[1, stride_y, stride_x, 1], padding=padding, name=name)
    conv = tf.nn.bias_add(conv, b)
    conv = tf.nn.relu(conv)

    return conv

def lrn(input_tensor, depth_radius, alpha, beta, bias=1.0):
    return tf.nn.local_response_normalization(input_tensor, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta)

def max_pool(input_tensor, ksize_y, ksize_x, stride_y, stride_x, padding, name):
    return tf.nn.max_pool(input_tensor, ksize=[1, ksize_y, ksize_x, 1], strides=[1, stride_y, stride_x, 1], padding=padding, name=name)

def fc(input_tensor, in_fc, out_fc, name, is_relu=True):
    w = tf.Variable(tf.random_normal([in_fc, out_fc], mean=0.0, stddev=0.01))
    b = tf.Variable(tf.zeros([out_fc]))

    fc = tf.nn.xw_plus_b(input_tensor, w, b)

    if is_relu == True:
        fc = tf.nn.relu(fc)

    return fc

class AlexNetModel(object):
    def __init__(self, learning_rate=0.01):

        tf.set_random_seed(1234)

        # Input Layer
        with tf.name_scope("input") as scope:

            self.dropout = tf.placeholder(tf.float32, name="dropout")

            # Input CIFAR-100 image size, 32x32x3
            # Color Channel 3(RGB)
            self.input_x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="input_x")

            # Augmentation
            # #1, Random Crop and Horizontal Reflection
            # #2, Altering RGB Intensities
            augmented_input_x = augment_image(self.input_x)

            # CIFAR-100 Label (0~99), one-hot vector
            self.input_y= tf.placeholder(tf.int32, [None, 1], name="input_y")

            input_y_one_hot = tf.one_hot(self.input_y, 100)
            input_y_one_hot = tf.reshape(input_y_one_hot, [-1, 100])

        with tf.name_scope("conv1") as scope:
                conv1 = conv(augmented_input_x, 11, 11, 3, 96, 4, 4, "VALID", "conv1")
                conv1 = lrn(conv1, 2.0, 10e-4, 0.75, 1.0)
                conv1 = max_pool(conv1, 3, 3, 2, 2, "VALID", "max_pool1")

        with tf.name_scope("conv2") as scope:
                conv2 = conv(conv1, 5, 5, 96, 256, 1, 1, "SAME", "conv2")
                conv2 = lrn(conv2, 2.0, 10e-4, 0.75, 1.0)
                conv2 = max_pool(conv2, 3, 3, 2, 2, "VALID", "max_pool2")

        with tf.name_scope("conv3") as scope:
                conv3 = conv(conv2, 3, 3, 256, 384, 1, 1, "SAME", "conv3")

        with tf.name_scope("conv4") as scope:
                conv4 = conv(conv3, 3, 3, 384, 384, 1, 1, "SAME", "conv4")

        with tf.name_scope("conv5") as scope:
                conv5 = conv(conv4, 3, 3, 384, 256, 1, 1, "SAME", "conv5")
                conv5 = max_pool(conv5, 3, 3, 2, 2, "VALID", "max_pool5")

        with tf.name_scope("fc6") as scope:
            flattened_size = 5 * 5 * 256  # 6 * 6 * 256
            flattened = tf.reshape(conv5, [-1, flattened_size])

            fc6 = fc(flattened, flattened_size, 4096, name="fc6")
            fc6 = tf.nn.dropout(fc6, keep_prob=self.dropout)

        with tf.name_scope("fc7") as scope:
            fc7 = fc(fc6, 4096, 4096, name="fc7")
            fc7 = tf.nn.dropout(fc7, keep_prob=self.dropout)

        with tf.name_scope("fc8") as scope:
            self.hypothesis = fc(fc7, 4096, 100, "fc8", False )

        # Loss Function
        with tf.name_scope("loss") as scope:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.hypothesis, labels=input_y_one_hot), name="loss")

        # Accuracy
        with tf.name_scope("accuracy") as scope:
            self.predictions = tf.argmax(self.hypothesis, 1, name="predictions")
            correct_predictions = tf.equal(self.predictions, tf.argmax(input_y_one_hot, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
