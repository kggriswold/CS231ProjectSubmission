import tensorflow as tf
import numpy as np
import math
from Layers import *

'''
def FCModel(inputShape, hiddenSizes, numClasses):
    inputSize = reduce(lambda x, y: x*y, inputShape)
    hiddenSizes.insert(0, inputSize)
    def genModel(inputs, keep_prob):
        inputs = tf.reshape(inputs,[-1,inputSize])
        currData = inputs
        for i in xrange(1, len(hiddenSizes)):
            with tf.name_scope('hidden%d' % (i)) as scope:
                currData = AffBatchReluDropLayer(currData, hiddenSizes[i-1],
                                                 hiddenSizes[i], keep_prob)
        with tf.name_scope('output') as scope:
            output = FCLayer(currData, hiddenSizes[-1], numClasses)
        return output
    return genModel
'''

def ConvModel(inputShape, convSizes, fcSizes, numClasses):
    convSizes.insert(0, inputShape[2])
    def genModel(inputs, keep_prob):
        currData = inputs
        batchNormList = []

        # Convolutional Layers
        numConv = 0
        numPool = 0
        for i in xrange(1, len(convSizes)):
            if (convSizes[i] == -1):
                with tf.name_scope('pool%d' % (numPool+1)) as scope:
                    currData = MaxPool2x2Layer(currData)
                convSizes[i] = convSizes[i-1]
                numPool += 1
            else:
                with tf.name_scope('conv%d' % (numConv+1)) as scope:
                    currData = ConvBatchReluDropLayer(
                        currData, convSizes[i-1], convSizes[i], 3, 3,
                        batchNormList, keep_prob)
                numConv += 1

        # Fully Connected Layers
        resultShape = [inputShape[0]/(2**numPool), inputShape[1]/(2**numPool),
                       convSizes[-1]]
        inputSize = reduce(lambda x, y: x*y, resultShape)
        fcSizes.insert(0, inputSize)

        currData = tf.reshape(currData,[-1,inputSize])
        for i in xrange(1, len(fcSizes)):
            with tf.name_scope('FC%d' % (i)) as scope:
                currData = AffBatchReluDropLayer(
                    currData, fcSizes[i-1], fcSizes[i], batchNormList,
                    keep_prob)
        with tf.name_scope('output') as scope:
            output = FCLayer(currData, fcSizes[-1], numClasses)
        return output, batchNormList
    return genModel

def ConvModelWithMax(inputShape, convSizes, fcSizes, numClasses):
    convSizes.insert(0, inputShape[2])
    def genModel(inputs, keep_prob):
        currData = inputs
        batchNormList = []

        # Convolutional Layers
        numConv = 0
        numPool = 0
        for i in xrange(1, len(convSizes)):
            if (convSizes[i] == -1):
                with tf.name_scope('pool%d' % (numPool+1)) as scope:
                    currData = MaxPool2x2Layer(currData)
                convSizes[i] = convSizes[i-1]
                numPool += 1
            else:
                with tf.name_scope('conv%d' % (numConv+1)) as scope:
                    currData = ConvBatchReluDropLayer(
                        currData, convSizes[i-1], convSizes[i], 3, 3,
                        batchNormList, keep_prob)
                numConv += 1

        # Fully Connected Layers
        fcSizes.insert(0, convSizes[-1])

        currData = tf.reduce_max(currData, reduction_indices=[1,2])
        for i in xrange(1, len(fcSizes)):
            with tf.name_scope('FC%d' % (i)) as scope:
                currData = AffBatchReluDropLayer(
                    currData, fcSizes[i-1], fcSizes[i], batchNormList,
                    keep_prob)
        with tf.name_scope('output') as scope:
            output = FCLayer(currData, fcSizes[-1], numClasses)
        return output, batchNormList
    return genModel

def ConvModelWithSplit(inputShape, convSizes, fcSizes, numClasses):
    convSizes.insert(0, inputShape[2])
    def genModel(inputs, keep_prob):
        currData = inputs
        batchNormList = []

        # Convolutional Layers
        numConv = 0
        numPool = 0
        for i in xrange(1, len(convSizes)):
            if (convSizes[i] == -1):
                with tf.name_scope('pool%d' % (numPool+1)) as scope:
                    currData = MaxPool2x2Layer(currData)
                convSizes[i] = convSizes[i-1]
                numPool += 1
            else:
                with tf.name_scope('conv%d' % (numConv+1)) as scope:
                    currData = ConvBatchReluDropLayer(
                        currData, convSizes[i-1], convSizes[i], 3, 1,
                        batchNormList, keep_prob)
                    currData = ConvBatchReluDropLayer(
                        currData, convSizes[i], convSizes[i], 1, 3,
                        batchNormList, keep_prob)
                numConv += 1

        # Fully Connected Layers
        resultShape = [inputShape[0]/(2**numPool), inputShape[1]/(2**numPool),
                       convSizes[-1]]
        inputSize = reduce(lambda x, y: x*y, resultShape)
        fcSizes.insert(0, inputSize)

        currData = tf.reshape(currData,[-1,inputSize])
        for i in xrange(1, len(fcSizes)):
            with tf.name_scope('FC%d' % (i)) as scope:
                currData = AffBatchReluDropLayer(
                    currData, fcSizes[i-1], fcSizes[i], batchNormList,
                    keep_prob)
        with tf.name_scope('output') as scope:
            output = FCLayer(currData, fcSizes[-1], numClasses)
        return output, batchNormList
    return genModel

# ConvModel1 was used to calibrate the batchSize, number of iterations, etc.
# for my computer's capabilities - it was not used as one of the experiments.
def ConvModel1(inputShape, numClasses):
    convSizes = [16, -1, 32, -1]
    fcSizes = [128]
    model = ConvModel(inputShape, convSizes, fcSizes, numClasses)
    return model

def ConvModel2(inputShape, numClasses):
    convSizes = [-1, 16, -1, 32, -1]
    fcSizes = [64]
    model = ConvModel(inputShape, convSizes, fcSizes, numClasses)
    return model

def ConvModel3(inputShape, numClasses):
    convSizes = [-1, 16, -1, 32, -1, -1, -1]
    fcSizes = [64]
    model = ConvModel(inputShape, convSizes, fcSizes, numClasses)
    return model

def ConvModel4(inputShape, numClasses):
    convSizes = [-1, 16, -1, 32]
    fcSizes = [32]
    model = ConvModelWithMax(inputShape, convSizes, fcSizes, numClasses)
    return model

def ConvModel5(inputShape, numClasses):
    convSizes = [-1, -1, 8, -1, 16, -1, -1]
    fcSizes = [32]
    model = ConvModel(inputShape, convSizes, fcSizes, numClasses)
    return model

def ConvModel6(inputShape, numClasses):
    convSizes = [-1, 16, -1, 32, -1]
    fcSizes = [64]
    model = ConvModelWithSplit(inputShape, convSizes, fcSizes, numClasses)
    return model

def ConvModel7(inputShape, numClasses):
    convSizes = [-1, 8, 8, -1, 16, 16, -1]
    fcSizes = [32, 32]
    model = ConvModelWithSplit(inputShape, convSizes, fcSizes, numClasses)
    return model

