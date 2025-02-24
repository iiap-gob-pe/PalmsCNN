from osgeo import gdal
import numpy as np
import os
import tensorflow as tf
from Palms_Segment.util_mod2 import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()
from sklearn.metrics import confusion_matrix, mean_squared_error,accuracy_score,cohen_kappa_score,classification_report,multilabel_confusion_matrix
import pandas as pd
import psutil


def printUseRam(titulo=""):
    print('\n'+titulo)
    print('Sytem RAM memory % used:', psutil.virtual_memory()[2])
    print('Sytem RAM memory GB used:', (psutil.virtual_memory()[3] / (1024 ** 2))/1024)
    print('Process use RAM GB: ',(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)/1024)
    print('')

def apply_semantic_segmentation_argmax(input_file_list,\
                                output_folder,\
                                test_model,\
                                window_radius,\
                                application_name='',
                                internal_window_radius=None,\
                                make_png=True,\
                                make_tif=True,\
                                local_scale_flag='none',\
                                global_scale_flag='none',\
                                png_dpi=200,\
                                verbose=True,
                                nodata_value=0,
                                response_file_list=[],
                                model_assessment=False,
                                scaling=True): #-9999
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
  
    feature_dim = gdal.Open(input_file_list[0],gdal.GA_ReadOnly).RasterCount
  
    if (os.path.isdir(output_folder) == False): os.mkdir(output_folder)
  
    if (internal_window_radius is None): internal_window_radius = window_radius

    for _item in range(0,len(input_file_list)):
      #for f in input_file_list:
      output_tif_file = os.path.join(output_folder,os.path.basename(input_file_list[_item]).split('.')[0] + '_' + application_name + '_argmax.tif')
      output_png_file = os.path.join(output_folder,os.path.basename(input_file_list[_item]).split('.')[0] + '_' + application_name + '_argmax.png')
      
      if(verbose): print(input_file_list[_item])

      
      dataset = gdal.Open(input_file_list[_item],gdal.GA_ReadOnly)
      if dataset.RasterCount>=4:
        bandas = 3
      else:
        bandas = dataset.RasterCount

      feature = np.zeros((dataset.RasterYSize,dataset.RasterXSize,bandas))
        
      for n in range(0,bandas):
        feature[:,:,n] = dataset.GetRasterBand(n+1).ReadAsArray()
        #print(feature[100,100,n] )SUSAN
  
      if (not dataset.GetRasterBand(1).GetNoDataValue() is None):
        feature[feature == dataset.GetRasterBand(1).GetNoDataValue()] = nodata_value
      feature[np.isnan(feature)] = nodata_value
      feature[np.isinf(feature)] = nodata_value
  
      '''
      if (global_scale_flag != 'none'):
       for n in range(0,feature.shape[2]):
        gd = feature[:,:,n] != nodata_value
        feature_scaling = scale_image(feature[:,:,n],global_scale_flag,nd=nodata_value)
        feature[gd,n] = feature[gd,n] - feature_scaling[0]
        feature[gd,n] = feature[gd,n] / feature_scaling[1]
      '''
      n_classes = test_model.predict(np.zeros((1,window_radius*2,window_radius*2,feature.shape[-1]))).shape[-1]
      
      output = np.zeros((feature.shape[0],feature.shape[1]))+ nodata_value
   
      cr = [0,feature.shape[1]]
      rr = [0,feature.shape[0]]
      
      collist = [x for x in range(cr[0]+window_radius,cr[1]-window_radius,internal_window_radius*2)]
      collist.append(cr[1]-window_radius)
      rowlist = [x for x in range(rr[0]+window_radius,rr[1]-window_radius,internal_window_radius*2)]
      rowlist.append(rr[1]-window_radius)
      
      
      for col in collist:
        if(verbose): print((col,cr[1]))
        images = []
        for n in rowlist:
          d = feature[n-window_radius:n+window_radius,col-window_radius:col+window_radius].copy()
          if(d.shape[0] == window_radius*2 and d.shape[1] == window_radius*2):
            #d = scale_image(d,local_scale_flag)
            #d = fill_nearest_neighbor(d)
            images.append(d)
        images = np.stack(images)
        images = images.reshape((images.shape[0],images.shape[1],images.shape[2],bandas))
        images = images.astype('float32')       
        #print(test_model.summary())
        pred_y = test_model.predict(images)
        _i = 0
        for n in rowlist:
          p = np.argmax(pred_y[_i,...].squeeze(),-1)
          #print("p.shape",p.shape)
          p[p==0] = 0 #background
          p[p==1] = 1 #M flexuosa
          p[p==2] = 2 #E precatoria
          p[p==3] = 3 #O bataua
          '''
		  p[p==0] = 0 #background
          p[p==1] = 50 #M flexuosa
          p[p==2] = 100 #E precatoria
          p[p==3] = 150 #O bataua
          '''
          if (internal_window_radius < window_radius):
            mm = rint(window_radius - internal_window_radius)            
            p = p[mm:-mm,mm:-mm]
          output[n-internal_window_radius:n+internal_window_radius,col-internal_window_radius:col+internal_window_radius] = p
          _i += 1
          if (_i >= len(images)):
            break
      
      output[feature[:,:,0] == nodata_value] = nodata_value
      
      #print(test_model.summary())
      if(verbose): print(output.shape) 
      if (make_tif):
        driver = gdal.GetDriverByName('GTiff') 
        driver.Register()
        output[np.isnan(output)] = nodata_value
         
        outDataset = driver.Create(output_tif_file,output.shape[1],output.shape[0],1,gdal.GDT_Float32)
        outDataset.SetProjection(dataset.GetProjection())
        outDataset.SetGeoTransform(dataset.GetGeoTransform())
        outDataset.GetRasterBand(1).WriteArray(output[:,:],0,0)
		
        del outDataset
      del dataset
      
      if model_assessment:
        nodata_value=0
        if(len(response_file_list)>0 & _item<=(len(response_file_list)-1)):
          print(response_file_list[_item])
          y_true = gdal.Open(response_file_list[_item],gdal.GA_ReadOnly)
          Iarray_true = y_true.GetRasterBand(1).ReadAsArray().astype(float)
          Iarray_true[Iarray_true==y_true.GetRasterBand(1).GetNoDataValue()]=nodata_value
          Iarray_true[np.isnan(Iarray_true)] = nodata_value
          Y_True=Iarray_true.flatten() #check size
          print("Y_True.shape",Y_True.shape)

          y_pred = gdal.Open(output_tif_file,gdal.GA_ReadOnly) 
          Iarray_pred = y_pred.GetRasterBand(1).ReadAsArray()
          Y_Pred=Iarray_pred.flatten() #check size
          print("Y_Pred.shape",Y_Pred.shape)


          # Generate confusion matrix
          
          cmatrix =pd.DataFrame(confusion_matrix(Y_True, Y_Pred),
					index = ["Background", "MFlexuosa", "EPrecatoria", "OBataua"],
					columns = ["Background", "MFlexuosa", "EPrecatoria", "OBataua"])
		  
          '''
          cmatrix =multilabel_confusion_matrix(Y_True, Y_Pred)
          '''
          target_names =["Background", "MFlexuosa", "EPrecatoria", "OBataua"]
          
          logs = open(output_tif_file+".logs", "w")
          print('Classification Report', file=logs)
          print('\n', file=logs)
          print('Confusion Matrix:', file=logs)
          print(cmatrix, file=logs)
          print('\n', file=logs)
          print(classification_report(Y_True, Y_Pred,target_names=target_names,digits=4),file=logs)
          logs.close()
    return os.path.basename(output_tif_file)


def apply_semantic_segmentation_argmax_optimizado(input_file_list,\
                                output_folder,\
                                model,\
                                window_radius,\
                                application_name='',
                                internal_window_radius=None,\
                                make_png=True,\
                                make_tif=True,\
                                local_scale_flag='none',\
                                global_scale_flag='none',\
                                png_dpi=200,\
                                verbose=True,
                                nodata_value=0,
                                response_file_list=[],
                                model_assessment=False,
                                scaling=True): #-9999
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
    printUseRam("========= Inicio de prediccion ==========") 
    win_size=window_radius * 2
  
    if (os.path.isdir(output_folder) == False): os.mkdir(output_folder)
  
    if (internal_window_radius is None): internal_window_radius = window_radius

    for _item in range(0,len(input_file_list)):
      #for f in input_file_list:
      output_tif_file = os.path.join(output_folder,os.path.basename(input_file_list[_item]).split('.')[0] + '_' + application_name + '_argmax.tif')
      output_png_file = os.path.join(output_folder,os.path.basename(input_file_list[_item]).split('.')[0] + '_' + application_name + '_argmax.png')
      
      if(verbose): print(input_file_list[_item])

      
      dataset = gdal.Open(input_file_list[_item],gdal.GA_ReadOnly)
      if dataset.RasterCount>=4:
        bandas = 3
      else:
        bandas = dataset.RasterCount     

      output = np.zeros((dataset.RasterYSize,dataset.RasterXSize))+ nodata_value
   
      cr = [0,dataset.RasterXSize]
      rr = [0,dataset.RasterYSize]


      collist = [x for x in range(cr[0]+window_radius,cr[1]-window_radius,internal_window_radius*2)]
      collist.append(cr[1]-window_radius)
      rowlist = [x for x in range(rr[0]+window_radius,rr[1]-window_radius,internal_window_radius*2)]
      rowlist.append(rr[1]-window_radius)
      printUseRam("Ante de aplicar corte y prediccion") 
      for col in collist:
          if(verbose): print((col,cr[1]))
          images = []
          for n in rowlist:
              d = np.zeros((win_size,win_size,bandas))
              for b in range(0,bandas):                  
                  d[:,:,b] = dataset.GetRasterBand(b+1).ReadAsArray(xoff=col-window_radius, yoff=n-window_radius,win_xsize=win_size, win_ysize=win_size)
                  d[np.isnan(d)] = nodata_value
                  d[np.isinf(d)] = nodata_value    
              if(d.shape[0] == window_radius*2 and d.shape[1] == window_radius*2):
                  images.append(d)
          images = np.stack(images)

          images = images.reshape((images.shape[0],images.shape[1],images.shape[2],bandas))
          ##Scalardatos
          if scaling:
            images=images / 127.5 - 1 
        
          pred_y = model.predict(images)
          
          _i = 0
          for n in rowlist:
            p = np.argmax(pred_y[_i,...].squeeze(),-1)
            p[p==0] = 0 #background
            p[p==1] = 1 #M flexuosa
            p[p==2] = 2 #E precatoria
            p[p==3] = 3 #O bataua
         
            if (internal_window_radius < window_radius):
              mm = rint(window_radius - internal_window_radius)            
              p = p[mm:-mm,mm:-mm]
            output[n-internal_window_radius:n+internal_window_radius,col-internal_window_radius:col+internal_window_radius] = p
            _i += 1
            if (_i >= len(images)):
              break
          printUseRam('Despues prediccion col '+str(col))      
      
      printUseRam('Despues de aplicar corte y prediccion')     
      #output[feature[:,:,0] == nodata_value] = nodata_value
      #del feature
      
      print('np.unique(output)',np.unique(output))
      if(verbose): print(output.shape) 
      if (make_tif):
        driver = gdal.GetDriverByName('GTiff') 
        driver.Register()
        output[np.isnan(output)] = nodata_value
         
        outDataset = driver.Create(output_tif_file,output.shape[1],output.shape[0],1,gdal.GDT_Float32)
        outDataset.SetProjection(dataset.GetProjection())
        outDataset.SetGeoTransform(dataset.GetGeoTransform())
        outDataset.GetRasterBand(1).WriteArray(output[:,:],0,0)
		
        del outDataset
      del dataset
      printUseRam('Despues de crear rater de prediccion')
      if model_assessment:
        nodata_value=0
        if(len(response_file_list)>0 & _item<=(len(response_file_list)-1)):
          print(response_file_list[_item])
          y_true = gdal.Open(response_file_list[_item],gdal.GA_ReadOnly)
          Iarray_true = y_true.GetRasterBand(1).ReadAsArray().astype(float)
          Iarray_true[Iarray_true==y_true.GetRasterBand(1).GetNoDataValue()]=nodata_value
          Iarray_true[np.isnan(Iarray_true)] = nodata_value
          Y_True=Iarray_true.flatten() #check size
          print("Y_True.shape",Y_True.shape)

          y_pred = gdal.Open(output_tif_file,gdal.GA_ReadOnly) 
          Iarray_pred = y_pred.GetRasterBand(1).ReadAsArray()
          Y_Pred=Iarray_pred.flatten() #check size
          print("Y_Pred.shape",Y_Pred.shape)


          # Generate confusion matrix
          
          cmatrix =pd.DataFrame(confusion_matrix(Y_True, Y_Pred),
					index = ["Background", "MFlexuosa", "EPrecatoria", "OBataua"],
					columns = ["Background", "MFlexuosa", "EPrecatoria", "OBataua"])
		  
          '''
          cmatrix =multilabel_confusion_matrix(Y_True, Y_Pred)
          '''
          target_names =["Background", "MFlexuosa", "EPrecatoria", "OBataua"]
          
          logs = open(output_tif_file+ application_name + ".logs", "w")
          print('Classification Report', file=logs)
          print('\n', file=logs)
          print('Confusion Matrix:', file=logs)
          print(cmatrix, file=logs)
          print('\n', file=logs)
          print(classification_report(Y_True, Y_Pred,target_names=target_names,digits=4),file=logs)
          logs.close()


 

