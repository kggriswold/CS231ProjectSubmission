import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import Model
import StateFarmDriverDetection.data as data
import time
import math
import random
import sys
import os

readDataTime = 0.0
trainTime = 0.0
batchNormTime = 0.0
evalTime = 0.0
testResultsTime = 0.0

def getXEntropyLoss(pred, golden):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(pred, golden,
                                                            name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_loss')
    return loss

def adamTrainer(lr=0.0001):
    def trainer(loss):
        optimizer = tf.train.AdamOptimizer(lr)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
    return trainer

def evalTopK(k=1):
    def evaluator(pred, golden):
        golden = tf.argmax(golden, 1)
        evalk = tf.nn.in_top_k(pred, golden, k)
        return tf.reduce_mean(tf.cast(evalk, tf.float32))
    return evaluator

def runDataSetMean(sess,
                   inputs,
                   golden,
                   dataHandler,
                   feedDict,
                   evaluators,
                   batchSize,
                   numEval,
                   useData='train'):
    if (useData != 'train' and useData != 'val'):
        print 'useData is %s - this is an invalid value' % (useData)
        sys.exit(1)

    numInput = dataHandler.getNumTrainImages()
    if (useData == 'val'):
        numInput = dataHandler.getNumValImages()
    if (numEval == -1 or numEval > numInput):
        numEval = numInput

    allIndicies = random.sample(range(numInput), numEval)
    results = []
    for i in xrange(0, numEval, batchSize):
        indicies = allIndicies[i:min(i+batchSize, numEval)]
        (images, labels) = (None, None)
        if (useData == 'train'):
            (images, labels) = dataHandler.readTrainData(indicies)
        else:
            (images, labels) = dataHandler.readValData(indicies)
            
        feedDict[inputs] = images
        feedDict[golden] = labels
        currRes = sess.run(evaluators, feed_dict = feedDict)
        for j in xrange(len(currRes)):
            currRes[j] *= len(indicies)
        results.append(currRes)
    resultTotal = results[0]
    for i in xrange(1,len(results)):
        for j in xrange(len(resultTotal)):
            resultTotal[j] += results[i][j]
    for j in xrange(len(resultTotal)):
        resultTotal[j] /= numEval
    return resultTotal

def initFeedDict(sess,
                 inputs,
                 golden,
                 keep_prob,
                 dataHandler,
                 batchNormList,
                 batchSize,
                 numEval,
                 folder='/dev/null'):
    global batchNormTime
    startTime = time.clock()

    feedDict = {}
    feedDict[keep_prob] = 1.0
    feedFileName = folder + '/batchNormVecs.npz'
    if (os.path.isfile(feedFileName)):
        print 'Reading in Batch Norm Vectors'
        f = open(feedFileName, 'r')
        batchNormVecs = np.load(f)
        for batchNormVec in batchNormList:
            feedDict[batchNormVec] = batchNormVecs[batchNormVec.name]
        f.close()
        print 'Finished Reading in Batch Norm Vectors'
    else:
        print 'Calculating Batch Norm Vectors'
        for batchNormVec in batchNormList:
            feedDict[batchNormVec] = runDataSetMean(
                sess, inputs, golden, dataHandler, feedDict, [batchNormVec],
                batchSize, numEval, useData='train')[0]
        print 'Finished Calculating Batch Norm Vectors'

        del feedDict[inputs]
        del feedDict[golden]
        if (os.path.isdir(folder)):
            print 'Saving Batch Norm Vectors'
            f = open(feedFileName, 'w')
            saveDict = dict((vec.name,arr) for vec,arr in feedDict.iteritems())
            np.savez(f, **saveDict)
            f.close()
            print 'Finished Saving Batch Norm Vectors'

    batchNormTime += time.clock() - startTime
    return feedDict
    

def getTrainValAcc(sess,
                   inputs,
                   golden,
                   keep_prob,
                   evaluators,
                   dataHandler,
                   batchNormList,
                   batchSize,
                   numEval,
                   folder='/dev/null'):
    global evalTime
    # Get the BatchNormalization vectors
    feedDict = initFeedDict(sess, inputs, golden, keep_prob, dataHandler,
                            batchNormList, batchSize, numEval, folder=folder)

    startTime = time.clock()
    # Evaluate the overall Training and Validation Accuracy
    trainRes = runDataSetMean(
        sess, inputs, golden, dataHandler, feedDict, evaluators, batchSize,
        numEval, useData='train')

    valRes = runDataSetMean(
        sess, inputs, golden, dataHandler, feedDict, evaluators, batchSize,
        numEval, useData='val')
    evalTime += time.clock() - startTime
    return trainRes, valRes

def printTestResults(sess,
                     inputs,
                     golden,
                     keep_prob,
                     pred,
                     dataHandler,
                     batchNormList,
                     batchSize,
                     folder='/dev/null'):
    global testResultsTime
    dataHandler.initTestData(folder+'/submission.csv')

    # Get the BatchNormalization vectors
    feedDict = initFeedDict(sess, inputs, golden, keep_prob, dataHandler,
                            batchNormList, batchSize, -1, folder=folder)

    startTime = time.clock()
    numInput = dataHandler.getNumTestImages()

    results = []
    for i in xrange(0, numInput, batchSize):
        print 'Image %d out of %d' % (i, numInput)
        indicies = range(i, min(i+batchSize, numInput))
        names, images = dataHandler.readTestData(indicies)
            
        feedDict[inputs] = images
        predictions = sess.run(pred, feed_dict = feedDict)

        dataHandler.writeTestData(names,predictions, folder+'/submission.csv')

    testResultsTime += time.clock() - startTime
    return

def runTraining(dataHandler,
                genModel,
                getLoss,
                trainer,
                evaluator,
                startIter = 0,
                numIters=2000,
                batchSize=100,
                dropoutProb=1.0,
                evalEvery=4000,
                numEval=2000,
                saveEvery=1000,
                folder='/dev/null',
                checkpoint=''):
    global readDataTime
    global trainTime
    inputShape = list(dataHandler.readTrainData([0])[0].shape)
    inputShape[0] = None
    numClasses = dataHandler.readTrainData([0])[1].shape[1]
    trainSize = dataHandler.getNumTrainImages()
    numBatches = int(math.ceil(float(trainSize) / batchSize))

    g = tf.Graph()
    losses = []
    with g.as_default():
        inputs = tf.placeholder(tf.float32,shape=inputShape)
        golden = tf.placeholder(tf.float32,shape=[None,numClasses])
        keep_prob = tf.placeholder(tf.float32)
        pred, batchNormList = genModel(inputs, keep_prob)

        loss = getLoss(pred, golden)
        train_op = trainer(loss)
        evaluate = evaluator(pred, golden)
        
        saver = tf.train.Saver()

        sess = tf.Session()
        
        init = tf.initialize_all_variables()
        sess.run(init)
        if (checkpoint != ''):
            saver.restore(sess, folder + checkpoint)
        
        for i in xrange(startIter, numIters):
            batchNum = (i*batchSize)/trainSize+1
            # Read a random selection of training examples
            newIndicies = random.sample(range(trainSize), batchSize)
            startTime = time.clock()
            (images, labels) = dataHandler.readTrainData(newIndicies)
            readDataTime += time.clock() - startTime
            
            # Train the network
            feedDict = {inputs: images, golden: labels,
                        keep_prob: dropoutProb}
            startTime = time.clock()
            _, currLoss = sess.run([train_op, loss], feed_dict = feedDict)
            trainTime += time.clock() - startTime
            losses.append(currLoss)
                
            print 'Iteration: %d, Batch: %d, Loss: %f' % (i+1,batchNum,currLoss)

            if ((i+1) % saveEvery == 0):
                saver.save(sess, folder + '/checkpoint', global_step=i+1)
                lossFile = open(folder + '/losses.txt', 'a')
                for l in losses:
                    lossFile.write(str(l) + '\n')
                lossFile.close()
                losses = []

            if ((i + 1) % evalEvery == 0):
                trainRes, valRes = getTrainValAcc(
                    sess, inputs, golden, keep_prob, [evaluate, loss],
                    dataHandler, batchNormList, batchSize, numEval)
                print 'Train Accuracy: %f' % trainRes[0]
                print 'Train Loss: %f' % trainRes[1]
                print 'Val Accuracy: %f' % valRes[0]
                print 'Val Loss: %f' % valRes[1]
                
        # Save the graph of the loss over time
        if (os.path.isdir(folder)):
            losses = []
            lossFile = open(folder + '/losses.txt', 'r')
            for line in lossFile:
                losses.append(float(line.rstrip()))
            lossFile.close()
            plt.plot(range(len(losses)), losses)
            plt.xlabel('Iteration')
            plt.ylabel('Cross Entropy Loss')
            plt.savefig(folder + '/loss-graph.png')

        # Print the test results
        printTestResults(sess, inputs, golden, keep_prob, tf.nn.softmax(pred),
                         dataHandler, batchNormList, batchSize, folder=folder)

        # Evaluate the final Performance
        trainRes, valRes = getTrainValAcc(
            sess, inputs, golden, keep_prob, [evaluate, loss], dataHandler,
            batchNormList, batchSize, -1, folder = folder)
        print 'Overall Train Accuracy: %f' % trainRes[0]
        print 'Overall Train Loss: %f' % trainRes[1]
        print 'Overall Val Accuracy: %f' % valRes[0]
        print 'Overall Val Loss: %f' % valRes[1]

        resultsFile = open(folder + '/results.txt', 'w')
        print >>resultsFile, 'Overall Train Accuracy: %f' % trainRes[0]
        print >>resultsFile, 'Overall Train Loss: %f' % trainRes[1]
        print >>resultsFile, 'Overall Val Accuracy: %f' % valRes[0]
        print >>resultsFile, 'Overall Val Loss: %f' % valRes[1]
        
        
startTotalTime = time.clock()

dataHandler = data.HandleData(shareDrivers=False)

inputShape = dataHandler.readTrainData([0])[0].shape[1:]
numClasses = dataHandler.readTrainData([0])[1].shape[1]

model = Model.ConvModel7(inputShape, numClasses)

runTraining(dataHandler, model, getXEntropyLoss, adamTrainer(lr=0.0001),
            evalTopK(k=1), startIter=0, numIters=2000, batchSize=50,
            dropoutProb=0.5, evalEvery=4000, numEval=2000, saveEvery=200,
            folder='StateFarmDriverDetection/sharedDriversModels/model7',
            checkpoint='')

totalTime = time.clock() - startTotalTime

print 'Time to Read In Data: %f' % (readDataTime)
print 'Time to Train: %f' % (trainTime)
print 'Time to Calculate Batch Norm Vectors: %f' % (batchNormTime)
print 'Time to Evaluate: %f' % (evalTime)
print 'Time to Calculate the Test Set results: %f' % (testResultsTime)
print 'Other Time: %f' % (totalTime - testResultsTime - evalTime - batchNormTime - trainTime - readDataTime)
print 'Total Time: %f' % (totalTime)
