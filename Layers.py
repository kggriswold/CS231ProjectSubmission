import tensorflow as tf
import numpy as np
import math

def FCLayer(inputs, inSize, outSize):
    with tf.name_scope('linear') as scope:
        weights = tf.Variable(tf.truncated_normal([inSize, outSize],
                              stddev=1.0 / math.sqrt(float(inSize))),
                              name='weights')
        biases = tf.Variable(tf.zeros([outSize]), name='biases')
        output = tf.matmul(inputs, weights) + biases
    return output

def Conv2DLayer(inputs, filter_height, filter_width, inSize, outSize):
    with tf.name_scope('conv2d_layer') as scope:
        weights = tf.Variable(
            tf.truncated_normal([filter_height, filter_width, inSize, outSize],
                                stddev=1.0 / math.sqrt(float(inSize))),
            name='weights')
        biases = tf.Variable(tf.zeros([outSize]), name='biases')
        output = tf.nn.conv2d(inputs, weights, [1,1,1,1], 'SAME') + biases
    return output

def MaxPool2x2Layer(inputs):
    with tf.name_scope('maxPool2x2_layer') as scope:
        output = tf.nn.max_pool(inputs, [1,2,2,1], [1,2,2,1], 'SAME')
    return output

def BatchNormLayer(inputs, batchNormList, dims=[0], epsilon=1e-12):
    with tf.name_scope('batch_norm') as scope:
        mean = tf.reduce_mean(inputs, dims, keep_dims=True)
        batchNormList.append(mean)
        shifted = inputs - mean
        squared = shifted**2
        variance = tf.reduce_mean(squared, dims, keep_dims=True)
        batchNormList.append(variance)
        normalized = tf.div(shifted, tf.sqrt(variance + epsilon))
        newVar = tf.Variable(1.0, name='new_variance')
        newMean = tf.Variable(0.0, name='new_mean')
        output = newVar*normalized + newMean
    return output

def AffBatchReluDropLayer(inputs, inSize, outSize, batchNormList, keep_prob):
    with tf.name_scope('AffBatchReluDrop') as scope:
        affine = FCLayer(inputs, inSize, outSize)
        batch = BatchNormLayer(affine, batchNormList, dims=[0])
        hidden = tf.nn.relu(batch)
        result = tf.nn.dropout(hidden, keep_prob)
    return result

def ConvBatchReluDropLayer(inputs, inSize, outSize, height,
                           width, batchNormList, keep_prob):
    with tf.name_scope('ConvBatchReluDrop') as scope:
        affine = Conv2DLayer(inputs, height, width, inSize, outSize)
        batch = BatchNormLayer(affine, batchNormList, dims=[0,1,2])
        hidden = tf.nn.relu(batch)
        result = tf.nn.dropout(hidden, keep_prob)
    return result
