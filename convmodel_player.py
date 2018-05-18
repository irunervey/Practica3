
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

n_input = 80 * 140

def dataSource(paths, batch_size):

    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image),  [float(i)]  # [one_hot(float(i), num_classes)]
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


with tf.variable_scope('ConvNet', reuse=False):
    x = tf.placeholder(tf.float32, [None, 80, 140, 1])

    o1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=3, activation=tf.nn.relu)
    o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
    o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
    o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

    h = tf.layers.dense(inputs=tf.reshape(o4, [1, 18*33*64]), units=20, activation=tf.nn.relu)
    y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)

# --------------------------------------------------
#
#       PLAY
#
# --------------------------------------------------

import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

cap = cv2.VideoCapture(0)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "./tmp/model.ckpt")
    print("Model restored.")

    while True:
        ret, img = cap.read()  # 720x1280x3 <-- print(img.shape);

        resized = cv2.resize(img, (140, 80), interpolation=cv2.INTER_AREA)
        #cropped = resized[0:180, 70:250]
        #resized64 = cv2.resize(cropped, (128, 128), interpolation=cv2.INTER_AREA)
        gray = np.asarray(cv2.cvtColor(resized, 7))

        cv2.imshow('Capture', gray)
        frame = gray.reshape(-1, 80, 140, 1)
        print (sess.run(y, feed_dict={x: frame}))

        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

cv2.destroyAllWindows()