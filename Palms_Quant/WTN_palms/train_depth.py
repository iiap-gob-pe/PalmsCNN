import depth_model
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

from post_process import *

VGG_MEAN = [103.939, 116.779, 123.68]

tf.set_random_seed(0)

### Select the nodes that will be used
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

def initialize_model(outputChannels, wd=None, modelWeightPaths=None):
    params = {"depth/conv1_1": {"name": "depth/conv1_1", "shape": [5,5,2,64], "std": None, "act": "relu", "reuse": False},
              "depth/conv1_2": {"name": "depth/conv1_2", "shape": [5,5,64,128], "std": None, "act": "relu", "reuse": False},
              "depth/conv2_1": {"name": "depth/conv2_1", "shape": [5,5,128,128], "std": None, "act": "relu", "reuse": False},
              "depth/conv2_2": {"name": "depth/conv2_2", "shape": [5,5,128,128], "std": None, "act": "relu", "reuse": False},
              "depth/conv2_3": {"name": "depth/conv2_3", "shape": [5,5,128,128], "std": None, "act": "relu", "reuse": False},
              "depth/conv2_4": {"name": "depth/conv2_4", "shape": [5,5,128,128], "std": None, "act": "relu", "reuse": False},
              "depth/fcn1": {"name": "depth/fcn1", "shape": [1,1,128,128], "std": None, "act": "relu", "reuse": False},
              "depth/fcn2": {"name": "depth/fcn2", "shape": [1,1,128,outputChannels], "std": None, "act": "relu", "reuse": False},
              "depth/upscore": {"name": "depth/upscore", "ksize": 8, "stride": 4, "outputChannels": outputChannels},
              }

    return depth_model.Network(params, wd=wd, modelWeightPaths=modelWeightPaths)

def forward_model(model, feeder, outputSavePath):
    with tf.Session() as sess:
        tfBatchDirs = tf.placeholder("float")
        tfBatchSS = tf.placeholder("float")
        keepProb = tf.placeholder("float")

        with tf.name_scope("model_builder"):
            print("attempting to build model")
            model.build(tfBatchDirs, tfBatchSS, keepProb=keepProb)
            print("built the model")

        init = tf.initialize_all_variables()

        sess.run(init)

        if not os.path.exists(outputSavePath):
            os.makedirs(outputSavePath)

        for i in range(int(math.floor(feeder.total_samples() / batchSize))):
            dirBatch, ssBatch, idBatch,ssMaskBatch = feeder.next_batch()

            outputBatch = sess.run(model.outputDataArgMax, feed_dict={tfBatchDirs: dirBatch,
                                                                      tfBatchSS: ssBatch,
                                                                      keepProb: 1.0})
            outputBatch = outputBatch.astype(np.uint8)

            for j in range(len(idBatch)):
                outputFilePath = os.path.join(outputSavePath, idBatch[j]+'.mat')
                outputFilePathImg = os.path.join(outputSavePath, idBatch[j])
                outputFileDir = os.path.dirname(outputFilePath)
                # print outputFileDir
                # print outputFilePath
                # raw_input("pause")

                if not os.path.exists(outputFileDir):
                    os.makedirs(outputFileDir)

                outputImage = watershed_cut(outputBatch[j], ssMaskBatch[j])
                skio.imsave(outputFilePathImg+'_watershed.jpg', scipy.ndimage.interpolation.zoom(outputImage, 1.0, mode='nearest', order=0))
                print('outputBatch[j].shape',outputBatch[j].shape)
                sio.savemat(outputFilePath, {"depth_map": outputBatch[j]})
                skio.imsave(outputFilePathImg+'_depth_map.jpg', scipy.ndimage.interpolation.zoom(outputBatch[j]*100, 1.0, mode='nearest', order=0))

                print("processed image %d out of %d"%(j+batchSize*i, feeder.total_samples()))

def train_model(model, outputChannels, learningRate, trainFeeder, valFeeder, modelSavePath=None, savePrefix=None, initialIteration=1):
    with tf.Session() as sess:
        tfBatchDirs = tf.placeholder("float")
        tfBatchGT = tf.placeholder("float")
        tfBatchWeight = tf.placeholder("float")
        tfBatchSS = tf.placeholder("float")
        keepProb = tf.placeholder("float")

        with tf.name_scope("model_builder"):
            print("attempting to build model")
            model.build(tfBatchDirs, tfBatchSS, keepProb=keepProb)
            print("built the model")
        sys.stdout.flush()
        loss = lossFunction.modelTotalLoss(pred=model.outputData, gt=tfBatchGT, weight=tfBatchWeight, ss=tfBatchSS, outputChannels=outputChannels)
        numPredictedWeighted = lossFunction.countTotalWeighted(ss=tfBatchSS, weight=tfBatchWeight)
        numPredicted = lossFunction.countTotal(ss=tfBatchSS)
        numCorrect = lossFunction.countCorrect(pred=model.outputData, gt=tfBatchGT, ss=tfBatchSS, k=1, outputChannels=outputChannels)

        train_op = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss=loss)

        init = tf.initialize_all_variables()

        sess.run(init)
        iteration = initialIteration
        dateejec=time.strftime(r"%d%m%Y%H%M%S")
        #saver = tf.train.Saver()
        while iteration < 500:
            batchLosses = []
            totalPredicted = 0
            totalCorrect = 0
            totalPredictedWeighted = 0

            for k in range(int(math.floor(valFeeder.total_samples() / batchSize))):
                dirBatch, gtBatch, weightBatch, ssBatch, _ = valFeeder.next_batch()
                # batchLoss, batchDepthError, batchPredicted, batchExceed3, batchExceed5 = \
                #     sess.run([loss, depthError, numPredicted, exceed3, exceed5],
                #                             feed_dict={tfBatchDirs: dirBatch,
                #                                         tfBatchGT: gtBatch,
                #                                         tfBatchSS: ssBatch})

                batchLoss, batchPredicted, batchPredictedWeighted, batchCorrect = sess.run([loss, numPredicted, numPredictedWeighted, numCorrect],
                    feed_dict={tfBatchDirs: dirBatch,
                               tfBatchGT: gtBatch,
                               tfBatchWeight: weightBatch,
                               tfBatchSS: ssBatch,
                               keepProb: 1.0})

                batchLosses.append(batchLoss)
                totalPredicted += batchPredicted
                totalPredictedWeighted += batchPredictedWeighted
                totalCorrect += batchCorrect
                
                #saver = tf.train.Saver() 
                #saver.save(sess, '../model/depth/logs/'+dateejec+'_'+str(iteration)+'-'+str(k)+'_logs-val-model')
            
            #saver.save(sess, '../model/depth/logs/'+dateejec+'_'+str(iteration)+'_logs-val-model')
            if np.isnan(np.mean(batchLosses)):
                print("LOSS RETURNED NaN")
                sys.stdout.flush()
                return 1

            # print "Itr: %d - b %d - val loss: %.3f, depth MSE: %.3f, exceed3: %.3f, exceed5: %.3f"%(iteration,j,
            #     float(np.mean(batchLosses)), totalDepthError/totalPredicted,
            #     totalExceed3/totalPredicted, totalExceed5/totalPredicted)
            print("%s Itr: %d - val loss: %.6f, correct: %.6f" % (time.strftime("%H:%M:%S"),
            iteration, float(np.mean(batchLosses)), totalCorrect / totalPredicted))

            if (iteration > 0 and iteration % 5 == 0) or checkSaveFlag(modelSavePath):
                modelSaver(sess, modelSavePath, savePrefix, iteration)

                # print "Processed iteration %d, batch %d" % (i,j)
                # sys.stdout.flush()

            sys.stdout.flush()
            # raw_input("paused")
            #for j in range(10):
            for j in range(int(math.floor(trainFeeder.total_samples() / batchSize))):
                dirBatch, gtBatch, weightBatch, ssBatch, _ = trainFeeder.next_batch()
                sess.run(train_op, feed_dict={tfBatchDirs: dirBatch,
                                              tfBatchGT: gtBatch,
                                              tfBatchWeight: weightBatch,
                                              tfBatchSS: ssBatch,
                                              keepProb: 0.7})
                ##Logs
                #print('print(loss)',loss)
                #saver = tf.train.Saver()
                #saver.save(sess, '../model/depth/logs/'+dateejec+'_'+str(iteration)+'-'+str(j)+'_logs-train-model')
            
            iteration += 1
        #saver.save(sess, '../model/depth/logs/'+dateejec+'_logs-train-model')


def modelSaver(sess, modelSavePath, savePrefix, iteration, maxToKeep=5):
    allWeights = {}

    for name in [n.name for n in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]:
        param = sess.run(name)
        nameParts = re.split('[:/]', name)
        saveName = nameParts[-4]+'/'+nameParts[-3]+'/'+nameParts[-2]
        allWeights[saveName] = param

        # print "Name: %s Mean: %.3f Max: %.3f Min: %.3f std: %.3f" % (name,
        #                                                              param.mean(),
        #                                                              param.max(),
        #                                                              param.min(),
        #                                                              param.std())
        # if name == "depth/fcn2/weights:0":
        #     for j in range(outputChannels):
        #         print "ch: %d, max %e, min %e, std %e" % (
        #             j, param[:, :, :, j].max(), param[:, :, :, j].min(), param[:, :, :, j].std())

    # raw_input("done")

    sio.savemat(os.path.join(modelSavePath, savePrefix+'_%03d'%iteration), allWeights)


def checkSaveFlag(modelSavePath):
    flagPath = os.path.join(modelSavePath, 'saveme.flag')

    if os.path.exists(flagPath):
        return True
    else:
        return False

if __name__ == "__main__":
    outputChannels = 16
    classType = 'predict_dir_onliclases_v4'
    indices = [0,1,2,3,4,5,6,7]
    # 0=car, 1=person, 2=rider, 3=motorcycle, 4=bicycle, 5=truck, 6=bus, 7=train
    savePrefix = "depth_" + classType + "_CR_pretrain"
    pathnpz="../dataset"
    outputPrefix = "../model/depth"

    train = True

    if train:
        batchSize = 6
        #learningRate = 1e-3
        #wd = 1e-6
        #wd = 1e-3
        learningRate = 1e-3#1e-4
        wd = 1e-6

        modelWeightPaths = []
        modelWeightPaths = [outputPrefix+'/depth_predict_dir_onliclases_v4_CR_pretrain_045.mat']
        initialIteration = 46

        model = initialize_model(outputChannels=outputChannels, wd=wd, modelWeightPaths=modelWeightPaths)
        trainFeeder = Batch_Feeder(dataset="cityscapes",
                                           indices=indices,
                                           train=train,
                                           batchSize=batchSize,
                                           padWidth=None,
                                           padHeight=None, flip=True, keepEmpty=False)
        trainFeeder.set_paths(idList=read_ids(pathnpz+'/Trainlist_filter_v3.txt'),
                         gtDir='../output/direction_ss/',
                         #gtDir=pathnpz+'/',
                         ssDir=pathnpz+'/')

        valFeeder = Batch_Feeder(dataset="cityscapes", indices=indices, train=train, batchSize=batchSize, padWidth=None, padHeight=None)
        valFeeder.set_paths(idList=read_ids(pathnpz+'/Vallist_filter_v3.txt'),
                         gtDir='../output/direction_ss/',
                         #gtDir=pathnpz+'/',
                         ssDir=pathnpz+'/')

        train_model(model=model, outputChannels=outputChannels,
                    learningRate=learningRate,
                    trainFeeder=trainFeeder,
                    valFeeder=valFeeder,
                    modelSavePath=outputPrefix,
                    savePrefix=savePrefix,
                    initialIteration=initialIteration)

    else:
        batchSize = 1
        modelWeightPaths = [outputPrefix+'/depth_predict_dir_onliclases_v4_CR_pretrain_045.mat']
        model = initialize_model(outputChannels=outputChannels, wd=0, modelWeightPaths=modelWeightPaths)
  
        feeder = Batch_Feeder(dataset="cityscapes", train=train, indices=indices, batchSize=batchSize, padWidth=None, padHeight=None)
        feeder.set_paths(idList=read_ids(pathnpz+'/Test_filter_v3.txt'),
                            gtDir=pathnpz+'',
                            ssDir=pathnpz+'')

        forward_model(model, feeder=feeder,
                      outputSavePath="../output/depth" % ())
