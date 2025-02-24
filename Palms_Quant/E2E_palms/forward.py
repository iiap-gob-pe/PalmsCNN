import tensorflow as tf
import os
import math
import sys
import scipy.io as sio
import skimage.io as skio
import scipy.ndimage.interpolation
from post_process import *
from io_utils import *
import numpy as np


def forward_model(model, feeder, outputSavePath, batchSize=1):
    with tf.Session() as sess:
        tfBatchImages = tf.placeholder("float")
        tfBatchSS = tf.placeholder("float")
        tfBatchSSMask = tf.placeholder("float")
        keepProb = tf.placeholder("float")

        with tf.name_scope("model_builder"):
            print("attempting to build model")
            model.build(tfBatchImages, tfBatchSS, tfBatchSSMask, keepProb=keepProb)
            print("built the model")

        init = tf.initialize_all_variables()

        sess.run(init)

        if not os.path.exists(outputSavePath):
            os.makedirs(outputSavePath)

        for i in range(int(math.floor(feeder.total_samples() / batchSize))):
            imageBatch, ssBatch, ssMaskBatch, idBatch = feeder.next_batch()
            print("imageBatch",imageBatch.shape)
            outputBatch = sess.run(model.outputDataArgMax, feed_dict={tfBatchImages: imageBatch,
                                                                      tfBatchSS: ssBatch,
                                                                      tfBatchSSMask: ssMaskBatch,
                                                                      keepProb: 1.0})

            #import matplotlib.pyplot as plt
            print("outputBatch",outputBatch.shape)
            #plt.imshow(outputBatch[0,:,:])
            #plt.show()
            
            outputBatch = outputBatch.astype(np.uint8)

            for j in range(len(idBatch)):
                outputFilePath = os.path.join(outputSavePath, idBatch[j] + '.png')
                outputFilePathgt = os.path.join(outputSavePath, idBatch[j] + '_gt.png')
                outputFilePathMat = os.path.join(outputSavePath, idBatch[j] + '.mat')
                outputFileDir = os.path.dirname(outputFilePath)

                if not os.path.exists(outputFileDir):
                    os.makedirs(outputFileDir)

                outputImage = watershed_cut(outputBatch[j], ssMaskBatch[j])
                        
                skio.imsave(outputFilePath, scipy.ndimage.interpolation.zoom(outputImage, 1.0, mode='nearest', order=0))
                skio.imsave(outputFilePathgt, scipy.ndimage.interpolation.zoom(outputBatch[j]*100, 1.0, mode='nearest', order=0))

                #sio.savemat(outputFilePathMat, {"depth_map": outputBatch[j]}, do_compression=True)

                print("processed image %d out of %d" % (j + batchSize * i + 1, feeder.total_samples()))
                sys.stdout.flush()

def forward_model_tiled(model,imageBatch,ssImage):
    with tf.Session() as sess:
        tfBatchImages = tf.placeholder("float")
        tfBatchSS = tf.placeholder("float")
        tfBatchSSMask = tf.placeholder("float")
        keepProb = tf.placeholder("float")
        with tf.name_scope("model_builder"):
            print("attempting to build model")
            model.build(tfBatchImages, tfBatchSS, tfBatchSSMask, keepProb=keepProb)
            print("built the model")

        init = tf.initialize_all_variables()

        sess.run(init)

        ssBatch, ssMaskBatch =ssProcess(ssImage[:,:,0])
        ssBatch=np.array([ssBatch])
        ssMaskBatch=np.array([ssMaskBatch])
        outputBatch = sess.run(model.outputDataArgMax, feed_dict={tfBatchImages: imageBatch,
                                                                      tfBatchSS: ssBatch,
                                                                      tfBatchSSMask: ssMaskBatch,
                                                                      keepProb: 1.0})
        outputImage = watershed_cut(outputBatch[0], ssMaskBatch[0])
        return outputImage