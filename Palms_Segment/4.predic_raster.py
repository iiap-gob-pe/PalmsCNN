##### Script to detect 3 palm species using Deeplabv3 ####
"""
Script to generate the predictions of 3 palm species using a trained Deeplabv3 Model for Palm Species Detection

This script is designed to generate predictions for three palm species in Amazonian forests using a trained Deeplabv3 model.
The model uses drone-based imagery as input and the Deeplabv3 architecture to detect the palms.

Classes: 4

Background (class 0)
Mauritia Flexuosa (class 1)
Euterpe Precatoria (class 2)
Oenocarpus Bataua (class 3)


This code belongs to:
Tagle et al. (2024).'Overcoming the Research-Implementation Gap through Drone-based Mapping of Economically Important Amazonian Palms'

The code leverages architectures from the repository:
- https://github.com/bonlime/keras-deeplab-v3-plus


Requirements:
- Python 3.x
- OpenCV


Authors: Tagle,X.; Cardenas, R.; Palacios, S.; Marcos, D.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

### Import libraries needed
from osgeo import gdal, ogr, osr #windows
#import gdal, ogr, osr #ALOBO
#import gdal #ALOBO
#import ogr  #ALOBO
#import osr  #ALOBO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
 
from PIL import Image
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import os
from tensorflow import keras
#from keras.models import model_from_json
from tensorflow.keras.models import load_model,model_from_json
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

#Import our modules
from Palms_Segment import model
from Palms_Segment import apply_model as apply_model_dl
from Palms_Quant.E2E_palms import apply_model as apply_model_dwt
from Palms_Segment import evaluate_data

import imageio
import cv2
import re
from Palms_Quant.E2E_palms.network_init import get_model
from tensorflow.keras import backend as kb
warnings.filterwarnings('ignore') #avoid getting warnings


#when CuDNN issue
import tensorflow as tf
#config = tf.ConfigProto()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
#session = tf.Session(config=config)


### Select the nodes that will be used
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

	
###Model settings
window_radius = 256
output_folder='output_test/All'
data_folder='data'
application_name='balanced'
#feature_file_list= [
#	'/data/PIU-03_1_5.tif',
#    '../../ecoCNN/data_prediction/20231219_142620_66_24c4_3B_AnalyticMS_SR.tif',
#	]

feature_file_list= [
        '../data/DMM-02_1.tif',
    '../data_prediction/Test_20250410.tif',
        ]

internal_window_radius = int(round(window_radius*0.75))


###Load model
#test_model = load_model('/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/deeplab_keras_model.h5',custom_objects={'relu6':model.relu6,'BilinearUpsampling':model.BilinearUpsampling })
#test_model = load_model('models/deeplab_keras_model_palms_iaa_all_0.003_W.h5',custom_objects={'relu6':model.relu6,'BilinearUpsampling':model.BilinearUpsampling,'dice_coef':model.dice_coef  })
#test_model = load_model('models/deeplab_keras_model_palms_class_balanced_RESCALADO-SINAUG_500EP_ft-iaaall0.003W_0.003_EP99.h5',custom_objects={'relu6':model.relu6,'BilinearUpsampling':model.BilinearUpsampling,'dice_coef':model.dice_coef  }) #ALOBO
#test_model = load_model('models/deeplab_keras_model_palms_all.h5',custom_objects={'relu6':model.relu6,'BilinearUpsampling':model.BilinearUpsampling })
#test_model = load_model('/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/models/deeplab_keras_model_palms_iaa_all_0.003_W.h5',custom_objects={'relu6':model.relu6,'BilinearUpsampling':model.BilinearUpsampling,'dice_coef':model.dice_coef  })
test_model = load_model('./models/deeplab_keras_model_palms_all_iaa_0.003.h5',custom_objects={'relu6':model.relu6,'BilinearUpsampling':model.BilinearUpsampling,'dice_coef':model.dice_coef  }) #ALOBO


###Model application

apply_model_dl.apply_semantic_segmentation_argmax_optimizado(
					input_file_list=feature_file_list,
                                        output_folder=output_folder,
					application_name=application_name,
                                        model=test_model,
                                        window_radius=window_radius,
                                        internal_window_radius=internal_window_radius,
                                        make_tif=True,
                                        make_png=True,
					scaling=True) 

###Computing the classification report #True when having ground truth for validation
#prediction_file_list=list(map(lambda x: os.path.join(output_folder,os.path.basename(x).split('.')[0] + '__argmax.tif'), feature_file_list))

"""
#response_file_list = ['../../ecoCNN/responses_raster/Palms_Clip_Brigida220622_1_classes.tif'] #y_true
#response_file_list = ['../../ecoCNN/responses_raster/Palms_merged_Sandoval31_07_1_classes.tif']
#response_file_list = ['../../ecoCNN/responses_raster/Palms_merged_DMM-02_1_classes.tif']
#response_file_list = ['../../ecoCNN/responses_raster/responses_raster/Palms_merged_Sandoval_Aguajal_1_test_2022_classes.tif']
##response_file_list = ['../../ecoCNN/responses_raster/responses_raster/Palms_merged_Elina210622_1_2022_classes.tif']
#response_file_list = ['../../ecoCNN/responses_raster/Palms_ALP-02_1_2_classes.tif'] #y_true
#response_file_list = ['../../ecoCNN/responses_raster/Palms_NJN-01_4_5_classes.tif'] #y_true
#response_file_list = ['../../ecoCNN/responses_raster/Palms_NJN-01_6_classes.tif'] #y_true
#response_file_list = ['../../ecoCNN/responses_raster/Palms_JHU-01_9_10_11_12_classes.tif'] #y_true
#response_file_list = ['../../ecoCNN/responses_raster/Palms_merged_SJR-01_1_classes.tif'] #y_true
#response_file_list = ['../../ecoCNN/responses_raster/Palms_merged_JEN-18_5_classes.tif'] #y_true
#response_file_list = ['../../ecoCNN/responses_raster/Palms_VEN-00_1_2_3_classes.tif'] #y_true
#response_file_list = ['../../ecoCNN/responses_raster/Palms_PIU-00_1_2_3_4_5_classes.tif'] #y_true
#response_file_list = ['../../ecoCNN/responses_raster/Palms_AGU-01_classes.tif'] #y_true

boundary_file_list=[]
#boundary_file_list = ['/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/SJR01_1_ROI_2019.shp']
#boundary_file_list = ['/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/JEN18_ROI_2018.shp']
#boundary_file_list = ['/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/DMM02_ROI_2019.shp']
#boundary_file_list = ['/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/JHU-01_ROI_2018.shp']
#boundary_file_list = ['/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/VEN00_ROI_2019.shp']
#boundary_file_list = ['/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/PIU00_ROI_2019.shp']
#boundary_file_list = ['/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/AGU01_ROI_2019.shp']
#boundary_file_list = ['/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/ALP02_ROI_2018.shp']
#boundary_file_list = ['/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/Brigida220622_1_ROI.shp']
#boundary_file_list = ['/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/Sandoval31_07_ROI_1.shp']
#boundary_file_list = ['/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/Sandoval_Aguajal_ROI.shp']				
##boundary_file_list = ['/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/Elina_ROI_22.shp']
#boundary_file_list = ['/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/NJN01_6_ROI_2019.shp']


prepare_data.model_assessment(prediction_file_list = prediction_file_list,
							response_file_list = response_file_list,
							response_vector_flag=False, #True when working with vectors (only one class)
							boundary_file_list=boundary_file_list, 
							boundary_file_vector_flag=True, #True when using ROIs
							application_name=application_name,
							output_folder = output_folder,
							ignore_projections=True
							)
"""
### End ###

