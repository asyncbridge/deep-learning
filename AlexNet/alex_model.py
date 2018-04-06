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

        num_of_gpu_use = 2
        conv2_group = []
        conv5_group = []
        fc6_group = []
        fc7_group = []

        for i in range(num_of_gpu_use):
            with tf.name_scope("conv1") as scope:
                with tf.device("/gpu:{}".format(i)):
                    conv1 = conv(augmented_input_x, 11, 11, 3, int(96 / num_of_gpu_use), 4, 4, "VALID", "conv{}".format(i))
                    conv1 = lrn(conv1, 2.0, 10e-4, 0.75, 1.0)
                    conv1 = max_pool(conv1, 3, 3, 2, 2, "VALID", "max_pool{}".format(i))

            with tf.name_scope("conv2") as scope:
                with tf.device("/gpu:{}".format(i)):
                    conv2 = conv(conv1, 5, 5, int(96 / num_of_gpu_use), int(256 / num_of_gpu_use), 1, 1, "SAME", "conv{}".format(i))
                    conv2 = lrn(conv2, 2.0, 10e-4, 0.75, 1.0)
                    conv2 = max_pool(conv2, 3, 3, 2, 2, "VALID", "max_pool{}".format(i))
                    conv2_group.append(conv2)

        # Communicate between gpu0-task and gpu1-task at certain layer
        merged_conv2 = tf.concat(conv2_group, 3)

        for i in range(num_of_gpu_use):
            with tf.name_scope("conv3") as scope:
                with tf.device("/gpu:{}".format(i)):
                    conv3 = conv(merged_conv2, 3, 3, 256, int(384 / num_of_gpu_use), 1, 1, "SAME", "conv{}".format(i))

            with tf.name_scope("conv4") as scope:
                with tf.device("/gpu:{}".format(i)):
                    conv4 = conv(conv3, 3, 3, int(384 / num_of_gpu_use), int(384 / num_of_gpu_use), 1, 1, "SAME", "conv{}".format(i))

            with tf.name_scope("conv5") as scope:
                with tf.device("/gpu:{}".format(i)):
                    conv5 = conv(conv4, 3, 3, int(384 / num_of_gpu_use), int(256 / num_of_gpu_use), 1, 1, "SAME", "conv{}".format(i))
                    conv5 = max_pool(conv5, 3, 3, 2, 2, "VALID", "max_pool{}".format(i))
                    conv5_group.append(conv5)

        # Communicate between gpu0-task and gpu1-task at certain layer
        merged_conv5 = tf.concat(conv5_group, 3)
        flattened_size = 5 * 5 * 256 # 6 * 6 * 256
        flattened = tf.reshape(merged_conv5, [-1, flattened_size])

        for i in range(num_of_gpu_use):
            with tf.name_scope("fc6") as scope:
                with tf.device("/gpu:{}".format(i)):
                    fc6 = fc(flattened, flattened_size, int(4096 / num_of_gpu_use), name="fc{}".format(i))
                    fc6 = tf.nn.dropout(fc6, keep_prob=self.dropout)
                    fc6_group.append(fc6)

        # Communicate between gpu0-task and gpu1-task at certain layer
        merged_fc6 = tf.concat(fc6_group, 1)

        for i in range(num_of_gpu_use):
            with tf.name_scope("fc7") as scope:
                with tf.device("/gpu:{}".format(i)):
                    fc7 = fc(merged_fc6, 4096, int(4096 / num_of_gpu_use), name="fc{}".format(i))
                    fc7 = tf.nn.dropout(fc7, keep_prob=self.dropout)
                    fc7_group.append(fc7)

         # Communicate between gpu0-task and gpu1-task at certain layer
        merged_fc7 = tf.concat(fc7_group, 1)

        with tf.name_scope("fc8") as scope:
            self.hypothesis = fc(merged_fc7, 4096, 100, "fc8", False )

        # Loss Function
        with tf.name_scope("loss") as scope:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.hypothesis, labels=input_y_one_hot), name="loss")

        # Accuracy
        with tf.name_scope("accuracy") as scope:
            self.predictions = tf.argmax(self.hypothesis, 1, name="predictions")
            correct_predictions = tf.equal(self.predictions, tf.argmax(input_y_one_hot, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
