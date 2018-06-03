import tensorflow as tf
import numpy as np

def TinyYOLOv2(features):
    mu = 0
    sigma = 1e-02

    # P0: Input ?x?x3 Output 416x416x3
    with tf.variable_scope("P0"):
        resized = tf.image.resize_images(features, (416, 416))
        print("P0: Input %s Output %s" % (features.get_shape(), resized.get_shape()))

    # C1: Input 416x416x3 Output 416x416x16
    with tf.variable_scope("C1"):
        W1 = tf.Variable(tf.truncated_normal(shape=(3, 3, 3, 16), mean=0, stddev=sigma))
        b1 = tf.Variable(tf.zeros(shape=(16)))
        C1 = tf.nn.conv2d(resized, W1, strides=(1, 1, 1, 1), padding="SAME")
        C1 = tf.add(C1, b1)
        C1 = tf.nn.relu(C1)
        print("C1: Input %s Output %s" % (resized.get_shape(), C1.get_shape()))
        C1 = tf.nn.dropout(C1, keep_prob=keep_prob)

    # S2: Input 416x416x16 Output 208x208x16
    with tf.variable_scope("S2"):
        S2 = tf.nn.max_pool(C1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")
        print("S2: Input %s Output %s" % (C1.get_shape(), S2.get_shape()))

    # C3: Input 208x208x16 Output 208x208x32
    with tf.variable_scope("C3"):
        W3 = tf.Variable(tf.truncated_normal(shape=(3, 3, 16, 32), mean=0, stddev=sigma))
        b3 = tf.Variable(tf.zeros(shape=(32)))
        C3 = tf.nn.conv2d(S2, W3, strides=(1, 1, 1, 1), padding="SAME")
        C3 = tf.add(C3, b3)
        C3 = tf.nn.relu(C3)
        print("C3: Input %s Output %s" % (S2.get_shape(), C3.get_shape()))
        C3 = tf.nn.dropout(C3, keep_prob=keep_prob)

    # S4: Input 208x208x32 Output 104x104x32
    with tf.variable_scope("S4"):
        S4 = tf.nn.max_pool(C3, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")
        print("S4: Input %s Output %s" % (C3.get_shape(), S4.get_shape()))

    # C5: Input 104x104x64 Output 104x104x64
    with tf.variable_scope("C5"):
        W5 = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 64), mean=0, stddev=sigma))
        b5 = tf.Variable(tf.zeros(shape=(64)))
        C5 = tf.nn.conv2d(S4, W5, strides=(1, 1, 1, 1), padding="SAME")
        C5 = tf.add(C5, b5)
        C5 = tf.nn.relu(C5)
        print("C1: Input %s Output %s" % (S4.get_shape(), C5.get_shape()))
        C5 = tf.nn.dropout(C5, keep_prob=keep_prob)

    # S6: Input 104x104x64 Output 52x52x64
    with tf.variable_scope("S6"):
        S6 = tf.nn.max_pool(C5, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")
        print("S6: Input %s Output %s" % (C5.get_shape(), S6.get_shape()))

    # C7: Input 52x52x64 Output 52x52x128
    with tf.variable_scope("C7"):
        W7 = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 128), mean=0, stddev=sigma))
        b7 = tf.Variable(tf.zeros(shape=(128)))
        C7 = tf.nn.conv2d(S6, W7, strides=(1, 1, 1, 1), padding="SAME")
        C7 = tf.add(C7, b7)
        C7 = tf.nn.relu(C7)
        print("C7: Input %s Output %s" % (S6.get_shape(), C7.get_shape()))
        C7 = tf.nn.dropout(C7, keep_prob=keep_prob)

    # S8: Input 52x52x128 Output 26x26x128
    with tf.variable_scope("S8"):
        S8 = tf.nn.max_pool(C7, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")
        print("S8: Input %s Output %s" % (C7.get_shape(), S8.get_shape()))

    # C9: Input 26x26x128 Output 26x26x256
    with tf.variable_scope("C9"):
        W9 = tf.Variable(tf.truncated_normal(shape=(3, 3, 128, 256), mean=0, stddev=sigma))
        b9 = tf.Variable(tf.zeros(shape=(256)))
        C9 = tf.nn.conv2d(S8, W9, strides=(1, 1, 1, 1), padding="SAME")
        C9 = tf.add(C9, b9)
        C9 = tf.nn.relu(C9)
        print("C9: Input %s Output %s" % (S8.get_shape(), C9.get_shape()))
        C9 = tf.nn.dropout(C9, keep_prob=keep_prob)

    # S10: Input 26x26x256 Output 13x13x256
    with tf.variable_scope("S10"):
        S10 = tf.nn.max_pool(C9, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")
        print("S10: Input %s Output %s" % (C9.get_shape(), S10.get_shape()))

    # C11: Input 13x13x256 Output 13x13x512
    with tf.variable_scope("C11"):
        W11 = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 512), mean=0, stddev=sigma))
        b11 = tf.Variable(tf.zeros(shape=(512)))
        C11 = tf.nn.conv2d(S10, W11, strides=(1, 1, 1, 1), padding="SAME")
        C11 = tf.add(C11, b11)
        C11 = tf.nn.relu(C11)
        print("C11: Input %s Output %s" % (S10.get_shape(), C11.get_shape()))
        C11 = tf.nn.dropout(C11, keep_prob=keep_prob)

    # S12: Input 13x13x512 Output 13x13x512
    with tf.variable_scope("S12"):
        S12 = tf.nn.max_pool(C11, ksize=(1, 2, 2, 1), strides=(1, 1, 1, 1), padding="SAME")
        print("S12: Input %s Output %s" % (C11.get_shape(), S12.get_shape()))

    # C13: Input 13x13x512 Output 13x13x1024
    with tf.variable_scope("C13"):
        W13 = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 1024), mean=0, stddev=sigma))
        b13 = tf.Variable(tf.zeros(shape=(1024)))
        C13 = tf.nn.conv2d(S12, W13, strides=(1, 1, 1, 1), padding="SAME")
        C13 = tf.add(C13, b13)
        C13 = tf.nn.relu(C13)
        print("C13: Input %s Output %s" % (S12.get_shape(), C13.get_shape()))
        C13 = tf.nn.dropout(C13, keep_prob=keep_prob)

    # C14: Input 13x13x1024 Output 13x13x512
    with tf.variable_scope("C14"):
        W14 = tf.Variable(tf.truncated_normal(shape=(3, 3, 1024, 512), mean=0, stddev=sigma))
        b14 = tf.Variable(tf.zeros(shape=(512)))
        C14 = tf.nn.conv2d(C13, W14, strides=(1, 1, 1, 1), padding="SAME")
        C14 = tf.add(C14, b14)
        C14 = tf.nn.relu(C14)
        print("C14: Input %s Output %s" % (C13.get_shape(), C14.get_shape()))
        C14 = tf.nn.dropout(C14, keep_prob=keep_prob)

    # C15: Input 13x13x512 Output 13x13x425
    with tf.variable_scope("C15"):
        W15 = tf.Variable(tf.truncated_normal(shape=(1, 1, 512, 425), mean=0, stddev=sigma))
        b15 = tf.Variable(tf.zeros(shape=(425)))
        C15 = tf.nn.conv2d(C14, W15, strides=(1, 1, 1, 1), padding="SAME")
        C15 = tf.add(C15, b15)
        C15 = tf.nn.relu(C15)
        print("C15: Input %s Output %s" % (C14.get_shape(), C15.get_shape()))
        C15 = tf.nn.dropout(C15, keep_prob=keep_prob)

    # NOTE: Final layer after C15 will darknet's detection layer
    return C15


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fname")
    args = parser.parse_args()

    features = tf.placeholder(tf.float32, shape=(None, None, None, 3))

    logits = TinyYOLOv2(features)
