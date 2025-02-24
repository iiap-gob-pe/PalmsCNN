##### Script to split segments (Palm crowns) ####
"""
Script to Train a Deep Watershed Model for Palm Species Quantification

This script is designed to train a model to split the crowns of three palm species in Amazonian forests
using drone-based imagery and the DWT architecture.

Classes: 4

Background (class 0)
Mauritia Flexuosa (class 1)
Euterpe Precatoria (class 2)
Oenocarpus Bataua (class 3)


This code belongs to:
Tagle et al. (2024).'Overcoming the Research-Implementation Gap through Drone-based Mapping of Economically Important Amazonian Palms'


The code leverages architectures from the repository:
- https://github.com/min2209/dwt


Requirements:
- Python 3.x
- OpenCV


Authors: Tagle,X.; Cardenas, R.; Palacios, S.; Marcos, D.
"""

from network_init import get_model
from io_utils import *
import tensorflow as tf
from forward import forward_model
from train import train_model
import os

### Select the nodes that will be used
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4"



### Parameter Initialization
tf.set_random_seed(0) # Seed for reproducibility


if __name__ == "__main__":
    outputChannels = 16
    savePrefix = "dwt_balanced_v1"
    outputPrefix = "../model/"
    #pathnpz="../output/e2e/tiles"
    pathnpz="../dataset"
    
    train = True
    tf.keras.backend.clear_session()
    if train:
        batchSize = 3
        learningRate = 5e-6 # usually 5e-6
        wd = 1e-6

        modelWeightPaths = None

        initialIteration = 1

        trainFeeder = Batch_Feeder(dataset="cityscapes", # 0=car, 1=person, 2=rider, 3=motorcycle, 4=bicycle, 5=truck, 6=bus, 7=train
                                           train=train,
                                           batchSize=batchSize,
                                           flip=True, keepEmpty=False, shuffle=True)

        trainFeeder.set_paths(idList=read_ids(pathnpz+'/Trainlist_balanced_classes.txt'),
                         imageDir=pathnpz+'/',
                         gtDir=pathnpz+'/',
                         #gtDir='../output/direction_ss/',
                         ssDir=pathnpz+'/')

        valFeeder = Batch_Feeder(dataset="cityscapes",
                                         train=train,
                                         batchSize=batchSize, shuffle=False)

        valFeeder.set_paths(idList=read_ids(pathnpz+'/Vallist_balanced_classes.txt'),
                         imageDir=pathnpz+'/',
                         gtDir=pathnpz+'/',
                         #gtDir='../output/direction_ss/',
                         ssDir=pathnpz+'/')

        model = get_model(wd=wd, modelWeightPaths=modelWeightPaths)

        train_model(model=model, outputChannels=outputChannels,
                    learningRate=learningRate,
                    trainFeeder=trainFeeder,
                    valFeeder=valFeeder,
                    modelSavePath=outputPrefix, 
                    savePrefix=savePrefix,
                    initialIteration=initialIteration)

    else:
        batchSize = 2
        modelWeightPaths = ["../model/dwt_onliclases_v5_045.mat"]

        model = get_model(modelWeightPaths=modelWeightPaths)

        feeder = Batch_Feeder(dataset="cityscapes",
                                      train=train,
                                      batchSize=batchSize)

        feeder.set_paths(idList=read_ids(pathnpz+'/test_npz.txt'),#Test_filter_v3.txt
                         imageDir=pathnpz+'/',
                            ssDir=pathnpz+'/')

        forward_model(model, feeder=feeder,
                      outputSavePath="../output/e2e")
