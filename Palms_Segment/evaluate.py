from osgeo import gdal
import numpy as np
import os

import tensorflow as tf
from Palms_Segment.util_mod2 import *
#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.gridspec as gridspec
#import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, mean_squared_error,accuracy_score,cohen_kappa_score,classification_report,multilabel_confusion_matrix
import pandas as pd

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="2,3"


def confusion(output_tif_file,output_folder,response_file_list): #-9999
    """ Apply a trained model to a series of files.
    
      Arguments:
      input_file_list - list
        List of feature files to apply the model to.
      output_folder - str
        Directory to place output images into.
      model - keras CNN model
        A pre-trained keras CNN model for semantic segmentation.
    
      Keyword Arguments:
      application_name - str
        A string to add into the output file name.
      internal_window_radius - int
        The size of the internal window on which to score the model.
      make_png - boolean
        Should an output be created in PNG format?
      make_tif - boolean
        Should an output be created in GeoTiff format?
      local_scale_flag - str
        A flag to apply local scaling (ie, scaling at the individual image level).
        Should match the local_scale_flage used to prepare training data.
        Options are:
          mean - mean center each image
          mean_std - mean center, and standard deviatio normalize each image
      global_scale_flag - str
        A flag to apply global scaling (ie, scaling at the level of input rasters).
      png_dpi - int
        The dpi of the generated PNG, if make_png is set to true.
      verbose - boolean
        An indication of whether or not to print outputs.
      nodata_value - float
        The value to set as the output nodata_value.
    
      Return:
      None, generates the specified output images + accuracy report.
    """
    nodata_value=0
    y_true = gdal.Open(response_file_list,gdal.GA_ReadOnly)
        #Iarray_true = y_true.GetRasterBand(1).ReadAsArray().astype(float)
    Iarray_true = y_true.GetRasterBand(1).ReadAsArray()
    Iarray_true[Iarray_true==y_true.GetRasterBand(1).GetNoDataValue()]=nodata_value
    Iarray_true[np.isnan(Iarray_true)] = nodata_value
    un_response_true = np.unique(Iarray_true[Iarray_true != nodata_value])
    print("True Response",un_response_true)
    Y_True=Iarray_true.flatten() #check size
    print("Y_True.shape",Y_True.shape)
    Ct = int(Iarray_true.max()) 
    #print (Ct)

    y_pred = gdal.Open(output_tif_file,gdal.GA_ReadOnly) 
    Iarray_pred = y_pred.GetRasterBand(1).ReadAsArray()
    Iarray_pred[Iarray_pred==y_true.GetRasterBand(1).GetNoDataValue()]=nodata_value
    Iarray_pred[np.isnan(Iarray_pred)] = nodata_value
    un_response_pre = np.unique(Iarray_pred[Iarray_pred!= nodata_value])
    print("Predict Response",un_response_pre)
    Y_Pred=Iarray_pred.flatten() #check size
    print("Y_Pred.shape",Y_Pred.shape)
    Cp = int(Iarray_pred.max()) 
    #print (Cp)
    if Ct>Cp:
        Cmax=Ct
    else:
        Cmax=Cp
    ind_array = np.arange(Cmax+1)
    names=['Background', 'MFlexuosa','EPrecatoria','OBataua']
    names = np.array(names)
    t_name= names [ind_array]
    #print(t_name)
    output_log_file = os.path.join(output_folder,os.path.basename(output_tif_file).split('.')[0] + '_argmax.logs')
        # Generate confusion matrix
          
    cmatrix =pd.DataFrame(confusion_matrix(Y_True, Y_Pred),index = t_name,columns = t_name)
    ##cmatrix =pd.DataFrame(confusion_matrix(Y_True, Y_Pred),index = ['Background', 'MFlexuosa','EPrecatoria'],columns = ['Background', 'MFlexuosa','EPrecatoria'])
		  
  #'EPrecatoria','OBataua'
   ## target_names =['Background', 'MFlexuosa','EPrecatoria']
          
    logs = open(output_log_file, "w")
    print('Classification Report', file=logs)
    print('\n', file=logs)
    print('Confusion Matrix:', file=logs)
    print(cmatrix, file=logs)
    print('\n', file=logs)
    print(classification_report(Y_True, Y_Pred,target_names=t_name,digits=4),file=logs)
    logs.close()

def confusion2(output,output_folder,response,file_name,application_name=''): #-9999
    """ Apply a trained model to a series of files.
    
      Arguments:
      input_file_list - list
        List of feature files to apply the model to.
      output_folder - str
        Directory to place output images into.
      model - keras CNN model
        A pre-trained keras CNN model for semantic segmentation.
    
      Keyword Arguments:
      application_name - str
        A string to add into the output file name.
      internal_window_radius - int
        The size of the internal window on which to score the model.
      make_png - boolean
        Should an output be created in PNG format?
      make_tif - boolean
        Should an output be created in GeoTiff format?
      local_scale_flag - str
        A flag to apply local scaling (ie, scaling at the individual image level).
        Should match the local_scale_flage used to prepare training data.
        Options are:
          mean - mean center each image
          mean_std - mean center, and standard deviatio normalize each image
      global_scale_flag - str
        A flag to apply global scaling (ie, scaling at the level of input rasters).
      png_dpi - int
        The dpi of the generated PNG, if make_png is set to true.
      verbose - boolean
        An indication of whether or not to print outputs.
      nodata_value - float
        The value to set as the output nodata_value.
    
      Return:
      None, generates the specified output images + accuracy report.
    """
    nodata_value=0
    #Iarray_true = y_true.GetRasterBand(1).ReadAsArray().astype(float)
    Iarray_true = response
    Iarray_true[np.isnan(Iarray_true)] = nodata_value
    un_response_true = np.unique(Iarray_true[Iarray_true != nodata_value])
    print("True Response",un_response_true)
    Y_True=Iarray_true.flatten() #check size
    print("Y_True.shape",Y_True.shape)
    Ct = int(Iarray_true.max()) 
    #print (Ct)

    Iarray_pred = output
    Iarray_pred[np.isnan(Iarray_pred)] = nodata_value
    un_response_pre = np.unique(Iarray_pred[Iarray_pred!= nodata_value])
    print("Predict Response",un_response_pre)
    Y_Pred=Iarray_pred.flatten() #check size
    print("Y_Pred.shape",Y_Pred.shape)
    Cp = int(Iarray_pred.max()) 
    print ("Cp",Cp)
    if Ct>Cp:
        Cmax=Ct
    else:
        Cmax=Cp
    #ind_array = np.arange(Cmax+1)
    ind_array=np.unique(Iarray_pred).astype(int)
    
    print('ind_array',ind_array)
    names=['Background', 'MFlexuosa','EPrecatoria','OBataua']
    names = np.array(names)
    t_name= names [ind_array]
    #print(t_name)
    output_log_file = os.path.join(output_folder,os.path.basename(file_name).split('.')[0] +'_'+application_name+ '_CM.logs')
    # Generate confusion matrix
    cmatrix =pd.DataFrame(confusion_matrix(Y_True, Y_Pred),index = t_name,columns = t_name)
    ##cmatrix =pd.DataFrame(confusion_matrix(Y_True, Y_Pred),index = ['Background', 'MFlexuosa','EPrecatoria'],columns = ['Background', 'MFlexuosa','EPrecatoria'])
		  
  #'EPrecatoria','OBataua'
   ## target_names =['Background', 'MFlexuosa','EPrecatoria']
          
    logs = open(output_log_file, "w")
    print('Classification Report', file=logs)
    print('\n', file=logs)
    print('Confusion Matrix:', file=logs)
    print(cmatrix, file=logs)
    print('\n', file=logs)
    print(classification_report(Y_True, Y_Pred,target_names=t_name,digits=4),file=logs)
    logs.close()


 

