import sys
import os
#sys.path.append('../code/Palms_Quant/E2E_palms')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from osgeo import gdal, ogr, osr
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, mean_squared_error,accuracy_score,cohen_kappa_score,classification_report,multilabel_confusion_matrix
import pandas as pd
import psutil
from io_utils import *
#from Palms_Quant.WTN_palms.train_depth import *
from util_dwt import *
from post_process import process_instances_raster,watershed_cut
import cv2
import re


def get_proj(fname,is_vector):
  """ Get the projection of a raster/vector dataset.
  Arguments:
  fname - str
    Name of input file.
  is_vector - boolean
    Boolean indication of whether the file is a vector or a raster.

  Returns:
  The projection of the input fname
  """
  if (is_vector):
    if (os.path.basename(fname).split('.')[-1] == 'shp'):
      vset = ogr.GetDriverByName('ESRI Shapefile').Open(fname,gdal.GA_ReadOnly)
    elif (os.path.basename(fname).split('.')[-1] == 'kml'):
      vset = ogr.GetDriverByName('KML').Open(fname,gdal.GA_ReadOnly)
    else:
      raise Exception('unsupported vector file type from file ' + fname)

    b_proj = vset.GetLayer().GetSpatialRef()
  else:
    b_proj = gdal.Open(fname,gdal.GA_ReadOnly).GetProjection()

  ##return re.sub('\W','',str(b_proj))
  srs = osr.SpatialReference()
  srs.ImportFromESRI([str(b_proj)])
  return re.sub('\W','',str(srs.ExportToProj4()))

def check_data_matches(set_a, set_b, set_b_is_vector=False,set_c=[],set_c_is_vector=True,ignore_projections=False):
    """ Check to see if two different gdal datasets have the same projection, geotransform, and extent.
    Arguments:
    set_a - list
      First list of gdal datasets to check.
    set_b - list
      Second list of gdal datasets (or vectors) to check.

    Keyword Arguments:
    set_b_is_vector - boolean
      Flag to indicate if set_b is a vector, as opposed to a gdal_dataset.
    set_c - list
      A third (optional) list of gdal datasets to check.
    set_c_is_vector - boolean
      Flag to indicate if set_c is a vector, as opposed to a gdal_dataset.
    ignore_projections - boolean
      A flag to ignore projection differences between feature and response sets - use only if you 
      are sure the projections are really the same.
      

    Return: 
    None, simply throw error if the check fails
    """
    if (len(set_a) != len(set_b)):
      raise Exception('different number of training features and responses')
    if (len(set_c) > 0):
      if (len(set_a) != len(set_c)):
        raise Exception('different number of training features and boundary files - give None for blank boundary')

    for n in range(0, len(set_a)):
      a_proj = get_proj(set_a[n],False)
      b_proj = get_proj(set_b[n],False)
   
      if (a_proj != b_proj and ignore_projections == False):
        raise Exception(('projection mismatch between', set_a[n], 'and', set_b[n],"prj_a ",a_proj,"prj_b",b_proj))

      if (len(set_c) > 0):
        if (set_c[n] is not None):
          print("File:",set_c[n])
          print("Pos:",n)
          c_proj = get_proj(set_c[n],set_c_is_vector)
        else:
          c_proj = b_proj

        if (a_proj != c_proj and ignore_projections == False):
          raise Exception(('projection mismatch between', set_a[n], 'and', set_c[n]))

      if (set_b_is_vector == False):
        dataset_a = gdal.Open(set_a[n],gdal.GA_ReadOnly)
        dataset_b = gdal.Open(set_b[n],gdal.GA_ReadOnly)

        ##if (dataset_a.GetProjection() != dataset_b.GetProjection() and ignore_projections == False):
            #raise Exception(('projection mismatch between', set_a[n], 'and', set_b[n]))

        if (dataset_a.GetGeoTransform() != dataset_b.GetGeoTransform()):
            raise Exception(('geotransform mismatch between', set_a[n], 'and', set_b[n],"GetGeoTransform = ",dataset_a.GetGeoTransform()," - ",dataset_b.GetGeoTransform()))

        if (dataset_a.RasterXSize != dataset_b.RasterXSize or dataset_a.RasterYSize != dataset_b.RasterYSize):
            raise Exception(('extent mismatch between', set_a[n], 'and', set_b[n]))

      if (len(set_c) > 0):
        if (set_c[n] is not None and set_c_is_vector == False):
          dataset_a = gdal.Open(set_a[n],gdal.GA_ReadOnly)
          dataset_c = gdal.Open(set_c[n],gdal.GA_ReadOnly)

          ##if (dataset_a.GetProjection() != dataset_c.GetProjection() and ignore_projections == False):
          ##    raise Exception(('projection mismatch between', set_a[n], 'and', set_c[n]))

          if (dataset_a.GetGeoTransform() != dataset_c.GetGeoTransform()):
              raise Exception(('geotransform mismatch between', set_a[n], 'and', set_c[n]))

          if (dataset_a.RasterXSize != dataset_c.RasterXSize or dataset_a.RasterYSize != dataset_c.RasterYSize):
              raise Exception(('extent mismatch between', set_a[n], 'and', set_c[n]))

def apply_instance_dwt_optimizado(feature_file_list,\
                                response_file_list,\
                                boundary_file_list,\
                                output_folder,\
                                model,\
                                window_radius,\
                                application_name='',
                                internal_window_radius=None,\
                                make_png=True,\
                                make_tif=True,\
                                local_scale_flag=None,\
                                global_scale_flag=None,\
                                png_dpi=200,\
                                verbose=False,
                                nodata_value= -9999,boundary_bad_value=0):
    """ Apply a trained model to a series of files.

      Arguments:
      feature_file_list - list
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
      None, simply generates the specified output images.
    """
    tf.compat.v1.disable_eager_execution()
    #printUseRam('Inicio de prediccion')
    check_data_matches(feature_file_list,response_file_list)


    if (os.path.isdir(output_folder) == False): os.mkdir(output_folder)

    if (internal_window_radius is None): internal_window_radius = window_radius

    win_size=window_radius * 2

    with tf.compat.v1.Session() as sess:
      tfBatchImages = tf.compat.v1.placeholder("float")
      tfBatchSS = tf.compat.v1.placeholder("float")
      tfBatchSSMask = tf.compat.v1.placeholder("float")
      keepProb = tf.compat.v1.placeholder("float")
      with tf.name_scope("model_builder"):
        #print("attempting to build model")
        model.build(tfBatchImages, tfBatchSS, tfBatchSSMask, keepProb=keepProb)
        #print("built the model")

      init = tf.compat.v1.global_variables_initializer()
      sess.run(init)

      for _i in range(0,len(feature_file_list)):
        output_tif_file = os.path.join(output_folder,os.path.basename(feature_file_list[_i]).split('.')[0] + '_' + application_name + '_dwt.tif')
        output_png_file = os.path.join(output_folder,os.path.basename(feature_file_list[_i]).split('.')[0] + '_' + application_name + '_dwt.png')


        dataset = gdal.Open(feature_file_list[_i],gdal.GA_ReadOnly)
        datasetresponse = gdal.Open(response_file_list[_i],gdal.GA_ReadOnly)

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

        ncollist=0
        for col in collist:
          if(verbose): print((col,cr[1]))
          imageBatch = []
          responses = []
          for n in rowlist:
            d = np.zeros((win_size,win_size,bandas))
            for b in range(0,bandas):
                  d[:,:,b] = dataset.GetRasterBand(b+1).ReadAsArray(xoff=col-window_radius, yoff=n-window_radius,win_xsize=win_size, win_ysize=win_size)
                  d[np.isnan(d)] = nodata_value
                  d[np.isinf(d)] = nodata_value
                  d[d==-9999] = nodata_value
            r = np.zeros((win_size,win_size))
            r[:,:] = datasetresponse.GetRasterBand(1).ReadAsArray(xoff=col-window_radius, yoff=n-window_radius,win_xsize=win_size, win_ysize=win_size).astype(float)

            if(d.shape[0] == window_radius*2 and d.shape[1] == window_radius*2):
              d = scale_image(d,local_scale_flag)
              # d = fill_nearest_neighbor(d)
              imageBatch.append(d)
              responses.append(r)
          imageBatch = np.stack(imageBatch)
          responses = np.stack(responses)

          ssBatch, ssMaskBatch = ssProcess_lists(responses)
          imageBatch = imageBatch.reshape((imageBatch.shape[0],imageBatch.shape[1],imageBatch.shape[2],bandas))

          nsplit=imageBatch.shape[0]//16 #20
          nsplit=1 if nsplit == 0 else nsplit
          s_imageBatch=np.array_split(imageBatch, nsplit)
          s_ssBatch=np.array_split(ssBatch, nsplit)
          s_ssMaskBatch=np.array_split(ssMaskBatch, nsplit)
          s_imageBatch=[x for x in s_imageBatch if x.size > 0]
          s_ssBatch=[x for x in s_ssBatch if x.size > 0]
          s_ssMaskBatch=[x for x in s_ssMaskBatch if x.size > 0]


          outputBatch = np.zeros((0,imageBatch.shape[1],imageBatch.shape[2]))
          for index in range(len(s_imageBatch)):
             tmp_outputBatch = sess.run(model.outputDataArgMax, feed_dict={tfBatchImages: s_imageBatch[index],
                                                                         tfBatchSS: s_ssBatch[index],
                                                                         tfBatchSSMask: s_ssMaskBatch[index],
                                                                         keepProb: 1.0})
             outputBatch=np.concatenate((outputBatch,tmp_outputBatch))


          outputBatch = outputBatch.astype(np.uint8)

          outputdwt = []
          for j in range(0,len(imageBatch)):

            outputImage = watershed_cut(outputBatch[j], ssMaskBatch[j])
            outputdwt.append(outputImage)
            tmask=ssMaskBatch[j]*100

          outputdwt = np.stack(outputdwt)


          _i = 0
          for n in rowlist:
            p = outputdwt[_i]
            if (internal_window_radius < window_radius):
              mm = int(np.rint(window_radius - internal_window_radius))
              p = p[mm:-mm,mm:-mm]
            output[n-internal_window_radius:n+internal_window_radius,col-internal_window_radius:col+internal_window_radius] = p
            _i += 1
            if (_i >= len(imageBatch)):
              break
          ncollist+=1
        output,quantification=process_instances_raster(output)

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

    return os.path.basename(output_tif_file),quantification['mauritia'],quantification['euterpe'],quantification['oenocarpus']
