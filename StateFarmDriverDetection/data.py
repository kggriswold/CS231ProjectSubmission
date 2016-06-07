import numpy as np
from scipy.ndimage import imread
import os
import random

class HandleData:
    def __init__(self, shareDrivers=False, folder = 'StateFarmDriverDetection',
                 trainingPercentage = 0.8):
        self.folder = folder

        # Read in the training metadata
        driverImgsFile = open('%s/driver_imgs_list.csv' % (self.folder), 'r')
        imageMetaData = {}
        sharedImageMetaData = []
        self.numClasses = 0
        numImages = 0
        for l in driverImgsFile:
            entries = l.split(',')
            if (entries[0] == 'subject'):
                continue
            if (entries[0] not in imageMetaData):
                imageMetaData[entries[0]] = []
            imageMetaData[entries[0]].append((entries[1], entries[2].rstrip()))

            sharedImageMetaData.append((entries[1], entries[2].rstrip()))

            classNum = int(entries[1].split('c')[1])
            if (classNum + 1 > self.numClasses):
                self.numClasses = classNum + 1

            numImages = numImages + 1
        driverImgsFile.close()

        # Read in the testing metadata
        self.testImageMetaData = os.listdir('%s/test' % (self.folder))

        # Shuffle the lists and split train and validation data
        random.seed(42)
        trainCutoff = int(round(trainingPercentage*numImages))
        self.trainImageMetaData = []
        self.valImageMetaData = []
        if (shareDrivers):
            sampled = random.sample(
                sharedImageMetaData,len(sharedImageMetaData))
            self.trainImageMetaData = sampled[:trainCutoff]
            self.valImageMetaData = sampled[trainCutoff:]
        else:
            sampled = random.sample(imageMetaData.keys(), len(imageMetaData))
            for i in xrange(len(sampled)):
                if (len(self.trainImageMetaData) <= trainCutoff):
                    self.trainImageMetaData.extend(imageMetaData[sampled[i]])
                else:
                    self.valImageMetaData.extend(imageMetaData[sampled[i]])

        random.shuffle(self.testImageMetaData)
        random.seed()

        numTrain = len(self.trainImageMetaData)
        numVal = len(self.valImageMetaData)
        print 'Number of Training Examples: %d' % numTrain
        print 'Number of Validation Examples: %d' % numVal
        print 'Training Percentage: %f'%(float(numTrain)/float(numTrain+numVal))
        

    def getNumTrainImages(self):
        return len(self.trainImageMetaData)

    def getNumValImages(self):
        return len(self.valImageMetaData)

    def getNumTestImages(self):
        return len(self.testImageMetaData)

    def readTrainData(self, indicies):
        return self.readTrainValData(self.trainImageMetaData, indicies)

    def readValData(self, indicies):
        return self.readTrainValData(self.valImageMetaData, indicies)

    def readTrainValData(self, metaData, indicies):
        requestedData = [metaData[i] for i in indicies]
        directory = '%s/train' % (self.folder)
        images = []
        labels = []
        
        for metaPoint in requestedData:
            image = imread('%s/%s/%s' % (directory, metaPoint[0], metaPoint[1]))
            labels.append(int(metaPoint[0].split('c')[1]))
            images.append(image)

        for i in xrange(len(labels)):
            l = [0.0 for _ in xrange(self.numClasses)]
            l[labels[i]] = 1.0
            labels[i] = l
   
        return (np.array(images), np.array(labels))

    def readTestData(self, indicies):
        requestedData = [self.testImageMetaData[i] for i in indicies]
        directory = '%s/test' % (self.folder)
        images = []
        names = []
        
        for metaPoint in requestedData:
            image = imread('%s/%s' % (directory, metaPoint))
            names.append(metaPoint)
            images.append(image)
   
        return (names, np.array(images))

    def initTestData(self, outFile):
        with open(outFile, 'w') as f:
            f.write('img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n')

    def writeTestData(self, names, predictions, outFile):
        with open(outFile, 'a') as f:
            for j in xrange(len(names)):
                line = names[j] + ','
                scores = ','.join(str(e) for e in
                                  np.ndarray.tolist(predictions[j]))
                f.write(line + scores + '\n')
