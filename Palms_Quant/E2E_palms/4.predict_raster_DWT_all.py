##### Script to detect the crowns of 3 palm species using DWT ####

### Import libraries needed
from network_init import get_model
from io_utils import *
import tensorflow as tf
# from forward import forward_model_tiled
from train import train_model
import os
import apply_model

### Select the nodes that will be used
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
##Guanabana
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" #0-3
##grenadilla
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4" #0-4

###Parameters for prediction
window_radius = 350
internal_window_radius = int(round(window_radius*0.75))

output_folder='../output/e2e/raster/test/All/dl_bdwt'

###Input
##Mosaic
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/VEN-01_01_02.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/VEN-01_01_02.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/dataset/test/Final/NJN-01_4_5_all.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/GAL-01_1.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/SNB-01.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/JEN-18_5.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/JEN-11_1_2_3_4_5.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/VEN-05_01.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/CIJH_VIV_1.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/SJR-01_6.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/NJN-01_6.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/PISC-01_5.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/QUI-01_14.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/VAP-03_1_2_3_4.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/CHE-03_3.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/VEN-01_09_10_11.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/VEN-02_8_9_10.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/VEN-03_9_10_11.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/VEN-04_06_7_8.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/VEN-05_08_9_10.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/JEN-15_10_11.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/AMA-01.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/VEN-04_06_7_8.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/PISC-02_9.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/KIN-01.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/SJR-01_1.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/POL-01_1.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/JHU-02_18_19.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/CHE-04_1_2_3.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/CIE-01_1.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/VEN-00_1_2_3.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/ALP-02_1_2.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/PIU-03_1_5.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/FDJH-01_1_2.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/JEN-14_04_5_6_7.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/VAP-02_1.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/JHU-01_9_10_11_12.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/VAP-01_1_2.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/NYO-00_1.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/SJO-00_01_2.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/SJO-01_01_2_3.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/CHE-04_1_2_3.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/CHE-05_1.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/HUA-01_1_2_3_4_5.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/MSH-01_1.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/SJR-01_6.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/PIU-03_1_5.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data/SAM-01_13.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/dataset/test/Final/Sandoval_Aguajal.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/dataset/test/Final/NJN-01_4_5_all.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data_prediction/ALP_20_0.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/dataset/test/Final/Clip_Sandoval31_07_1.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data_prediction/Clip_Sandoval_Aguajal_1.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data_prediction/Clip_Elina210622_1.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data_prediction/Clip_Brigida220622_1.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data_prediction/Clip_AM-02.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data_prediction/DMM-02_1.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data_prediction/Clip_DMM-01_1_2_3_4_5_10_11.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data_prediction/Clip_PRN-01_11_12_13.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data_prediction/Clip_PRN-01_11_12_13_TEST.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data_prediction/Clip_ALP-60_8_9_10_11_12_13_14_21_22.tif"] 
raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data_prediction/Clip_ALP-60_8_9_10_11_12_13_14_21_22_TEST.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data_prediction/Clip_PIU-02_9_13_14_15_16_17_TEST.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data_prediction/AGU-01.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data_prediction/NAR-01_1_2_3_4_5_all.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data_prediction/NYO-03_1_5_6_7_8_9_transparent_mosaic_group1.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data_prediction/VAP-01_1_2.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/secoCNN/data_prediction/PIU-00_1_2_3_4_5_transparent_mosaic_group1.tif"] 
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data_prediction/PIU-02_9_13_14_15_16_17_transparent_mosaic_group1.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data_prediction/Clip_VEN_0312_0313_0314_0315_0316_0317_0318_049_0410_0411_058_0511_0512_0513_modi.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data_prediction/VEN_total_1.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data_prediction/VEN_total_2.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data_prediction/VEA-01_1_transparent_mosaic_group1.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data_prediction/PISC-01_1.tif"]
#raster=["/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/data_prediction/Clip_VAP-01_1_2_TEST.tif"]

##Segmentation mask (output from deeplab or other)
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/R1/VEN-01_01_02__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/NJN-01_4_5_all__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/NJN-01_4_5_all_balanced_argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/VEN-00_1_2_3__argmax.tif"] 
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/ALP-02_1_2__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/CIJH_VIV_1__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/JEN-18_5__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/JEN-11_1_2_3_4_5__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/VEN-05_01__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/SJR-01_6__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/PISC-01_5__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/SNB-01__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/QUI-01_14__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/VAP-03_1_2_3_4__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/CHE-03_3__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/VEN-01_09_10_11__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/VEN-02_8_9_10__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/VEN-03_9_10_11__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/VEN-04_06_7_8__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/VEN-05_08_9_10__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/JEN-15_10_11__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/AMA-01__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/VEN-04_06_7_8__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/PISC-02_9__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/KIN-01__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/SJR-01_1__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/POL-01_1__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/JHU-02_18_19__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/GAL-01_1__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/CHE-04_1_2_3__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/CIE-01_1__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/Clip_AM-02__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/Clip_AM-02_balanced_argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/FDJH-01_1_2__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/PIU-03_1_5__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/JEN-14_04_5_6_7__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/VAP-02_1__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/JHU-01_9_10_11_12_balanced_argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/JHU-01_9_10_11_12__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/PIU-02_9_13_14_15_16_17_transparent_mosaic_group1__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/VAP-01_1_2_balanced_argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/VAP-01_1_2__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/NJN-01_6__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/NYO-00_1__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/SJO-00_01_2__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/SJO-01_01_2_3__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/CHE-04_1_2_3__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/CHE-05_1__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/MSH-01_1__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/SJR-01_6__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/PIU-03_1_5__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/SAM-01_13__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/HUA-01_1_2_3_4_5__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/PIU-00_1_2_3_4_5_transparent_mosaic_group1__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/Sandoval_Aguajal__argmax.tif"] 
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/NJN-01_4_5_all__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/ALP_20_0__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/Clip_Sandoval31_07_1__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/Clip_Sandoval31_07_1_balanced_argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/Clip_Sandoval_Aguajal_1_balanced_argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/Clip_Sandoval_Aguajal_1__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/Clip_Elina210622_1__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/Clip_Elina210622_1_balanced_argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/Clip_Brigida220622_1_balanced_argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/Clip_Brigida220622_1__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/Clip_AM-02_balanced_argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/DMM-02_1__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/Clip_DMM-01_1_2_3_4_5_10_11__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/Clip_PRN-01_11_12_13__argmax.tif"] 
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/Clip_PRN-01_11_12_13_TEST_balanced_argmax.tif"] 
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/Clip_PRN-01_11_12_13_TEST__argmax.tif"]
mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/Clip_ALP-60_8_9_10_11_12_13_14_21_22_TEST__argmax.tif"] 
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/Clip_ALP-60_8_9_10_11_12_13_14_21_22_TEST_balanced_argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/Clip_ALP-60_8_9_10_11_12_13_14_21_22__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/Clip_PIU-02_9_13_14_15_16_17_TEST_balanced_argmax.tif"] 
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/AGU-01__argmax.tif"] 
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/NAR-01_1_2_3_4_5_all_balanced_argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/NYO-03_1_5_6_7_8_9_transparent_mosaic_group1__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/VAP-01_1_2__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/Clip_VEN_0312_0313_0314_0315_0316_0317_0318_049_0410_0411_058_0511_0512_0513_modi__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/VEN_total_1__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/VEN_total_2__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/VEA-01_1_transparent_mosaic_group1_balanced_argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/VEA-01_1_transparent_mosaic_group1__argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/PISC-01_1_balanced_argmax.tif"]
#mask=["/mnt/guanabana/raid/home/xtagle/ML/CNN/deeplab/keras-deeplab-v3-plus/output_test/All/Clip_VAP-01_1_2_TEST_balanced_argmax.tif"]

##ROI (optional)
# roi=["../data/VEN05_ROI_2017.shp"]
roi=[]

#modelWeightPaths = ["../model/dwt_onliclases_v5_045.mat"] #dwt_palms_pspsnet_v2_030
modelWeightPaths = ["../model/dwt_balanced_v1_070.mat"] #dwt_palms_pspsnet_v2_030

model = get_model(modelWeightPaths=modelWeightPaths)

###Prediction settings
apply_model.apply_instance_dwt_optimizado(raster,
                                        mask,
                                        roi,
                                        output_folder,
                                        model,
                                        window_radius,
                                        internal_window_radius=internal_window_radius,
                                        make_tif=True,
                                        make_png=True)
####End####

