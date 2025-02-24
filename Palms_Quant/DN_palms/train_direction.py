##### Script to pre-train the Direction Network ####

### Import libraries needed
import direction_model
from ioUtils import *
import math
import lossFunction
import tensorflow as tf
import numpy as np
import sys
import os
import scipy.io as sio
import re
import time

import scipy.io as sio
import skimage.io as skio
import scipy.ndimage.interpolation


### Select the nodes that will be used
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,4"
#os.environ ['CUDA_VISIBLE_DEVICES'] = ''


VGG_MEAN = [103.939, 116.779, 123.68]

tf.set_random_seed(0)

def initialize_model(outputChannels, wd=None, modelWeightPaths=None):
    fuseChannels=256
    params = {"direction/conv1_1": {"name": "direction/conv1_1", "shape": [3,3,4,64], "std": None, "act": "relu"},
            "direction/conv1_2": {"name": "direction/conv1_2", "shape": [3,3,64,64], "std": None, "act": "relu"},
            "direction/conv2_1": {"name": "direction/conv2_1", "shape": [3,3,64,128], "std": None, "act": "relu"},
            "direction/conv2_2": {"name": "direction/conv2_2", "shape": [3,3,128,128], "std": None, "act": "relu"},
            "direction/conv3_1": {"name": "direction/conv3_1", "shape": [3,3,128,256], "std": None, "act": "relu"},
            "direction/conv3_2": {"name": "direction/conv3_2", "shape": [3,3,256,256], "std": None, "act": "relu"},
            "direction/conv3_3": {"name": "direction/conv3_3", "shape": [3,3,256,256], "std": None, "act": "relu"},
            "direction/conv4_1": {"name": "direction/conv4_1", "shape": [3,3,256,512], "std": None, "act": "relu"},
            "direction/conv4_2": {"name": "direction/conv4_2", "shape": [3,3,512,512], "std": None, "act": "relu"},
            "direction/conv4_3": {"name": "direction/conv4_3", "shape": [3,3,512,512], "std": None, "act": "relu"},
            "direction/conv5_1": {"name": "direction/conv5_1", "shape": [3,3,512,512], "std": None, "act": "relu"},
            "direction/conv5_2": {"name": "direction/conv5_2", "shape": [3,3,512,512], "std": None, "act": "relu"},
            "direction/conv5_3": {"name": "direction/conv5_3", "shape": [3,3,512,512], "std": None, "act": "relu"},
            "direction/fcn5_1": {"name": "direction/fcn5_1", "shape": [5,5,512,512], "std": None, "act": "relu"},
            "direction/fcn5_2": {"name": "direction/fcn5_2", "shape": [1,1,512,512], "std": None, "act": "relu"},
            "direction/fcn5_3": {"name": "direction/fcn5_3", "shape": [1,1,512,fuseChannels], "std": 1e-2, "act": "relu"},
            "direction/upscore5_3": {"name": "direction/upscore5_3", "ksize": 8, "stride": 4, "outputChannels": fuseChannels},
            "direction/fcn4_1": {"name": "direction/fcn4_1", "shape": [5,5,512,512], "std": None, "act": "relu"},
            "direction/fcn4_2": {"name": "direction/fcn4_2", "shape": [1,1,512,512], "std": None, "act": "relu"},
            "direction/fcn4_3": {"name": "direction/fcn4_3", "shape": [1,1,512,fuseChannels], "std": 1e-3, "act": "relu"},
            "direction/upscore4_3": {"name": "direction/upscore4_3", "ksize": 4, "stride": 2, "outputChannels": fuseChannels},
            "direction/fcn3_1": {"name": "direction/fcn3_1", "shape": [5,5,256,256], "std": None, "act": "relu"},
            "direction/fcn3_2": {"name": "direction/fcn3_2", "shape": [1,1,256,256], "std": None, "act": "relu"},
            "direction/fcn3_3": {"name": "direction/fcn3_3", "shape": [1,1,256,fuseChannels], "std": 1e-4, "act": "relu"},
            "direction/fuse3_1": {"name": "direction/fuse_1", "shape": [1,1,fuseChannels*3, 512], "std": None, "act": "relu"},
            "direction/fuse3_2": {"name": "direction/fuse_2", "shape": [1,1,512,512], "std": None, "act": "relu"},
            "direction/fuse3_3": {"name": "direction/fuse_3", "shape": [1,1,512,outputChannels], "std": None, "act": "lin"},
            "direction/upscore3_1": {"name": "direction/upscore3_1", "ksize": 8, "stride": 4, "outputChannels":outputChannels}}

    return direction_model.Network(params, wd=wd, modelWeightPaths=modelWeightPaths)

def forward_model(model, feeder, outputSavePath):
    with tf.Session() as sess:
        #tfBatchImages = tf.placeholder("float", shape=[None, 512, 1024, 3])
        #tfBatchSS = tf.placeholder("float", shape=[None, 512, 1024])
        #tfBatchSSMask = tf.placeholder("float", shape=[None, 512, 1024])

        tfBatchImages = tf.placeholder("float")
        tfBatchSS = tf.placeholder("float")
        tfBatchSSMask = tf.placeholder("float")

        with tf.name_scope("model_builder"):
            print("attempting to build model")
            model.build(tfBatchImages, tfBatchSS, tfBatchSSMask)
            print("built the model")
        sys.stdout.flush()

        init = tf.initialize_all_variables()
        sess.run(init)

        for i in range(int(math.floor(feeder.total_samples() / batchSize))):
            imageBatch, ssBatch, ssMaskBatch, idBatch = feeder.next_batch()

            outputBatch = sess.run(model.output, feed_dict={tfBatchImages: imageBatch, tfBatchSS: ssBatch, tfBatchSSMask: ssMaskBatch})

            for j in range(len(idBatch)):
                outputFilePath = os.path.join(outputSavePath, idBatch[j]+'.mat')
                outputFilePathImg = os.path.join(outputSavePath, idBatch[j])
                #outputFilePathNpz = os.path.join(outputSavePath+'/predict_for_train', idBatch[j])
                outputFilePathNpz = os.path.join(outputSavePath, idBatch[j])
                outputFileDir = os.path.dirname(outputFilePath)

                if not os.path.exists(outputFileDir):
                    os.makedirs(outputFileDir)
                print('outputBatch[j].shape',outputBatch[j].shape)
                #sio.savemat(outputFilePath, {"dir_map": outputBatch[j]}, do_compression=True)
                np.savez_compressed(outputFilePathNpz+'_dirmap', outputBatch[j])
                skio.imsave(outputFilePathImg+'_dirx.jpg', scipy.ndimage.interpolation.zoom(outputBatch[j][:,:,0], 1.0, mode='nearest', order=0))
                skio.imsave(outputFilePathImg+'_diry.jpg', scipy.ndimage.interpolation.zoom(outputBatch[j][:,:,1], 1.0, mode='nearest', order=0))

                print("processed image %d out of %d"%(j+batchSize*i, feeder.total_samples()))

def train_model(model, outputChannels, learningRate, trainFeeder, valFeeder, modelSavePath=None, savePrefix=None, initialIteration=1):
    with tf.Session() as sess:

        '''    
        tfBatchImages = tf.placeholder("float", shape=[None, 512, 512, 3])
        tfBatchGT = tf.placeholder("float", shape=[None, 512, 512, 2])
        tfBatchWeight = tf.placeholder("float", shape=[None, 512, 512])
        tfBatchSS = tf.placeholder("float", shape=[None, 512, 512])
        tfBatchSSMask = tf.placeholder("float", shape=[None, 512, 512])
        '''
        tfBatchImages = tf.placeholder("float")
        tfBatchGT = tf.placeholder("float")
        tfBatchWeight = tf.placeholder("float")
        tfBatchSS = tf.placeholder("float")
        tfBatchSSMask = tf.placeholder("float")
        
        with tf.name_scope("model_builder"):
            print("attempting to build model")
            model.build(tfBatchImages, tfBatchSS, tfBatchSSMask)
            print("built the model")

        sys.stdout.flush()
        
        loss = lossFunction.angularErrorLoss(pred=model.output, gt=tfBatchGT, weight=tfBatchWeight, ss=tfBatchSS, outputChannels=outputChannels)

        angleError = lossFunction.angularErrorTotal(pred=model.output, gt=tfBatchGT, weight=tfBatchWeight, ss=tfBatchSS, outputChannels=outputChannels)
        numPredicted = lossFunction.countTotal(ss=tfBatchSS)
        numPredictedWeighted = lossFunction.countTotalWeighted(ss=tfBatchSS, weight=tfBatchWeight)
        exceed45 = lossFunction.exceedingAngleThreshold(pred=model.output, gt=tfBatchGT,
                                                        ss=tfBatchSS, threshold=45.0, outputChannels=outputChannels)
        exceed225 = lossFunction.exceedingAngleThreshold(pred=model.output, gt=tfBatchGT,
                                                        ss=tfBatchSS, threshold=22.5, outputChannels=outputChannels)

        train_op = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss=loss)

        init = tf.initialize_all_variables()

        sess.run(init)
        iteration = initialIteration

        while iteration < 1000: #1000
            batchLosses = []
            totalAngleError = 0
            totalExceed45 = 0
            totalExceed225 = 0
            totalPredicted = 0
            totalPredictedWeighted = 0

            for k in range(int(math.floor(valFeeder.total_samples() / batchSize))):
                imageBatch, gtBatch, weightBatch, ssBatch, ssMaskBatch, _ = valFeeder.next_batch()

                batchLoss, batchAngleError, batchPredicted, batchPredictedWeighted, batchExceed45, batchExceed225 = sess.run(
                    [loss, angleError, numPredicted, numPredictedWeighted, exceed45, exceed225],
                    feed_dict={tfBatchImages: imageBatch,
                               tfBatchGT: gtBatch,
                               tfBatchWeight: weightBatch,
                               tfBatchSS: ssBatch,
                               tfBatchSSMask: ssMaskBatch})
                # print "ran iteration"
                batchLosses.append(batchLoss)
                totalAngleError += batchAngleError
                totalPredicted += batchPredicted
                totalPredictedWeighted += batchPredictedWeighted
                totalExceed45 += batchExceed45
                totalExceed225 += batchExceed225

            if np.isnan(np.mean(batchLosses)):
                print("LOSS RETURNED NaN")
                sys.stdout.flush()
                return 1

            print("%s Itr: %d - val loss: %.3f, angle MSE: %.3f, exceed45: %.3f, exceed22.5: %.3f" % (
                time.strftime("%H:%M:%S"), iteration,
            float(np.mean(batchLosses)), totalAngleError / totalPredictedWeighted,
            totalExceed45 / totalPredicted, totalExceed225 / totalPredicted))
            sys.stdout.flush()

            if (iteration > 0 and iteration % 5 == 0) or checkSaveFlag(modelSavePath):
                modelSaver(sess, modelSavePath, savePrefix, iteration)

            for j in range(int(math.floor(trainFeeder.total_samples() / batchSize))):
                # print "running batch %d"%(j)
                # sys.stdout.flush()
                imageBatch, gtBatch, weightBatch, ssBatch, ssMaskBatch, _ = trainFeeder.next_batch()
                sess.run(train_op, feed_dict={tfBatchImages: imageBatch,
                                              tfBatchGT: gtBatch,
                                              tfBatchWeight: weightBatch,
                                              tfBatchSS: ssBatch,
                                              tfBatchSSMask: ssMaskBatch})
            iteration += 1

def modelSaver(sess, modelSavePath, savePrefix, iteration, maxToKeep=5):
    allWeights = {}
    for name in [n.name for n in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]:
        param = sess.run(name)
        nameParts = re.split('[:/]', name)
        saveName = nameParts[-4]+'/'+nameParts[-3]+'/'+nameParts[-2]
        allWeights[saveName] = param

    weightsFileName = os.path.join(modelSavePath, savePrefix+'_%03d'%iteration)

    sio.savemat(weightsFileName, allWeights)


def checkSaveFlag(modelSavePath):
    flagPath = os.path.join(modelSavePath, 'saveme.flag')

    if os.path.exists(flagPath):
        return True
    else:
        return False


if __name__ == "__main__":
    outputChannels = 2
    classType = 'onliclases_v4'
    indices = [0,1,2,3]
    # 0=car, 1=person, 2=rider, 3=motorcycle, 4=bicycle, 5=truck, 6=bus, 7=train
    savePrefix = "direction_" + classType + ""
    train = False #True
    pathnpz="../dataset"
    outputPrefix = "../model/direction"

    if train:
        batchSize = 8 #4
        learningRate = 1e-4 #1e-5
        # learningRateActual = 1e-7
        wd = 1e-5 # weight decay 1e-5

        #modelWeightPaths = ["./cityscapes/models/direction/VGG16init_conv1_ch4.mat"]
        #modelWeightPaths=None
        modelWeightPaths=[outputPrefix+'/direction_onliclases_v4_025.mat'] #direction_onliclases_v2_495.mat
        initialIteration = 6

        model = initialize_model(outputChannels=outputChannels, wd=wd, modelWeightPaths=modelWeightPaths)

        trainFeeder = Batch_Feeder(dataset="cityscapes", indices=indices, train=train, batchSize=batchSize,
                                   padWidth=None, padHeight=None, flip=True, keepEmpty=False)
        trainFeeder.set_paths(idList=read_ids(pathnpz+'/Trainlist_filter_v3.txt'), #filter_trainlist_old.txt Trainlist_filter_v2
                         imageDir=pathnpz+'/',
                         gtDir=pathnpz+'/',
                         ssDir=pathnpz+'/')

        valFeeder = Batch_Feeder(dataset="cityscapes", indices=indices, train=train, batchSize=batchSize,
                                 padWidth=None, padHeight=None)

        valFeeder.set_paths(idList=read_ids(pathnpz+'/Vallist_filter_v3.txt'), #filter_vallist_old Vallist_filter_v2
                         imageDir=pathnpz+'/',
                         gtDir=pathnpz+'/',
                         ssDir=pathnpz+'/')

        train_model(model=model, outputChannels=outputChannels,
                    learningRate=learningRate,
                    trainFeeder=trainFeeder, valFeeder=valFeeder,
                    modelSavePath=outputPrefix, 
                    savePrefix=savePrefix,
                    initialIteration=initialIteration)
    else:
        batchSize = 1 #5
        modelWeightPaths=[outputPrefix+'/direction_onliclases_v4_300.mat'] 

        model = initialize_model(outputChannels=outputChannels, wd=0, modelWeightPaths=modelWeightPaths)

        feeder = Batch_Feeder(dataset="cityscapes", indices=indices, train=train, batchSize=batchSize, padWidth=None, padHeight=None)
        feeder.set_paths(idList=read_ids(pathnpz+'/Test_filter_v3.txt'),#all_filter_trainlist_old.txt Test_filter_v3.txt Data_filter_v3.txt
                         imageDir=pathnpz+'/',
                         ssDir=pathnpz+'/')

        forward_model(model, feeder=feeder,
                      outputSavePath="../output/direction_ss")
