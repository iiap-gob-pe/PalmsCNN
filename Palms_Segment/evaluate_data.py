##### Script to prepare data for the accuracy assessment ####

### Import libraries needed
from osgeo import gdal, ogr, osr
import numpy as np
import sys,os
import fiona
import rasterio.features
import re
import imageio
import cv2

import rasterio
import rasterio.mask
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling

from Palms_Segment.util_mod2 import *
from Palms_Segment import evaluate

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
  




def check_data_matches(set_a, set_b, set_b_is_vector=False,set_c=[],set_c_is_vector=False,ignore_projections=False):
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
      b_proj = get_proj(set_b[n],set_b_is_vector)
   
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




def rasterize_vector(vector_file,geotransform,output_shape):
    """ Rasterizes an input vector directly into a numpy array.

    Arguments:
    vector_file - str
      Input vector file to be rasterized.
    geotransform - list
      A gdal style geotransform.
    output_shape - tuple
      The shape of the output file to be generated.

    Return:
    A rasterized 2-d numpy array.
    """
    ds = fiona.open(vector_file,'r')
    geotransform = [geotransform[1],geotransform[2],geotransform[0],geotransform[4],geotransform[5],geotransform[3]]
    mask = np.zeros(output_shape)
    for n in range(0,len(ds)):
      rasterio.features.rasterize([ds[n]['geometry']],transform=geotransform,default_value=1,out=mask)
    return mask


def model_assessment(prediction_file_list, response_file_list, response_vector_flag=True, boundary_file_list = [], boundary_file_vector_flag = True, boundary_bad_value=0, ignore_projections=False, output_folder='',application_name=''): 

  """ Main externally called function, transforms a list of feature/response rasters into 
    a set of training data at a specified window size

    Arguments:
    window_radius - determines the subset image size, which results as 2*window_radius  
    samples_per_response_per_site - either an integer (used for all sites) or a list of integers (one per site)
                       that designates the maximum number of samples to be pulled per response 
                       from that location.  If the number of responses is less than the samples
                       per site, than the number of responses available is used
    prediction_file_list - file list of the feature rasters (mosaic)
    response_file_list - file list of the response rasters

    Keyword Arguments:
    response_vector_flag  - boolean
      A boolean indication of whether the response type is a vector or a raster (True for vector).
    boundary_file_list - list
      An optional list of boundary files for each feature/response file.
    boundary_file_vector_flag - boolean
      A boolean indication of whether the boundary file type is a vector or a raster (True for vector).
    
    verbose - boolean
      A flag indicating printout verbosity, set to True to get print outputs, False to have no printing.
    ignore_projections - boolean
      A flag to ignore projection differences between feature and response sets - use only if you 
      are sure the projections are really the same.

    Return: 
    Confusion matrix
          
  """
  #check_data_matches(prediction_file_list,response_file_list,response_vector_flag,boundary_file_list,boundary_file_vector_flag,ignore_projections)
  
  
  for _i in range(0,len(prediction_file_list)):
    
    with fiona.open(boundary_file_list[_i], "r") as shapefile:
        AOI = [feature["geometry"] for feature in shapefile]

    with rasterio.open(prediction_file_list[0]) as src:
        print('prediction',src.shape)
        P_clipped, out_P_clipped = rasterio.mask.mask(src, AOI, crop=True)

    with rasterio.open(response_file_list[0]) as src:
        print('response',src.shape)
        R_clipped, out_P_clipped = rasterio.mask.mask(src, AOI, crop=True)

    print('P_clipped.shape',P_clipped.shape)
    print('R_clipped.shape',R_clipped.shape)
    evaluate.confusion2(P_clipped[0],output_folder,R_clipped[0],prediction_file_list[0])
    
    return print("done")
    
    """
    if(verbose): print(feature.shape)
    # ensure nodata values are consistent 
    if (not dataset.GetRasterBand(1).GetNoDataValue() is None):
      feature[feature == dataset.GetRasterBand(1).GetNoDataValue()] = nodata_value
    feature[np.isnan(feature)] = nodata_value
    feature[np.isinf(feature)] = nodata_value
    response[feature[:,:,0] == nodata_value] = nodata_value
    #feature[response == nodata_value,:] = nodata_value
	"""
    
   
