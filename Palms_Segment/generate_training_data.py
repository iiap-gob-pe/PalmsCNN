##### Functions to generate the training data with 4 classes ####
"""
Script with functions to generate the training data to Train a Deeplabv3 Model for Palm Species Detection

This script is designed to generate training data to train a model to detect three palm species in Amazonian forests
using drone-based imagery and the Deeplabv3 architecture. It is a part of the study:
'Overcoming the Research-Implementation Gap through Drone-based Mapping of Economically Important Amazonian Palms'
by Tagle et al. (2024).

The code leverages architectures from the repository:
- https://github.com/pgbrodrick/ecoCNN


Requirements:
- Python 3.x
- OpenCV


Authors: Tagle,X.; Cardenas, R.; Palacios, S.; Marcos, D.
"""


### Import libraries needed
from osgeo import gdal, ogr, osr
import numpy as np
import sys,os
import fiona
import rasterio.features
import re
import imageio
import cv2

#from .util_mod2 import *util_mod2 import *
from .util_mod2 import * #ALOBO


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



def build_semantic_segmentation_training_data(window_radius, samples_per_response_per_site, feature_file_list, response_file_list, response_vector_flag=True, boundary_file_list = [], boundary_file_vector_flag = True, boundary_bad_value=0, internal_window_radius=None, center_random_offset_fraction=0.0, response_repeats=1,savename=None, nodata_maximum_fraction=0.5, response_minimum_fraction=0.0, fill_in_feature_data=True, global_scale_flag=None, local_scale_flag=None, nodata_value=0, random_seed=13, n_folds=10, verbose=False, ignore_projections=False,nclases=4): #nclases=False when go back to ecoCNN

  """ Main externally called function, transforms a list of feature/response rasters into 
    a set of training data at a specified window size

    Arguments:
    window_radius - determines the subset image size, which results as 2*window_radius  
    samples_per_response_per_site - either an integer (used for all sites) or a list of integers (one per site)
                       that designates the maximum number of samples to be pulled per response 
                       from that location.  If the number of responses is less than the samples
                       per site, than the number of responses available is used
    feature_file_list - file list of the feature rasters (mosaic)
    response_file_list - file list of the response rasters

    Keyword Arguments:
    response_vector_flag  - boolean
      A boolean indication of whether the response type is a vector or a raster (True for vector).
    boundary_file_list - list
      An optional list of boundary files for each feature/response file.
    boundary_file_vector_flag - boolean
      A boolean indication of whether the boundary file type is a vector or a raster (True for vector).
    internal_window_radius - int
      An inner image subset used to score the algorithm, and within which a response must lie to 
      be included in training data
    center_random_offset_fraction - float
      The fraction to randomly shuffle data from around response center.
    response_repeats - int 
      The number of times to re-caputre each response value from different offset fractions.
    savename - str
      The basename to save scaling and munged data, if None than nothing is saved.
    nodata_maximum_fraction - float
      The maximum fraction of nodata_values to allow in each training sample.
    response_minimum_fraction - float
      The minimum response fraction that must be in each training sample.
    fill_in_feature_data - boolean
      A flag to fill in missing data with a nearest neighbor interpolation.
    global_scale_flag - str
      A flag to apply global scaling (ie, scaling at the level of input rasters).
    local_scale_flag - str
      A flag to apply local scaling (ie, scaling at the individual image level).
      Options are:
        mean - mean center each image
        mean_std - mean center, and standard deviation normalize each image
    nodata_value - float
      The value to ignore from the feature or response dataset.
    random_seed - int
      A random seed to set (for reproducibility), set to None to not set a seed.
    n_folds - int
      The number of folds to set up for data training.
    verbose - boolean
      A flag indicating printout verbosity, set to True to get print outputs, False to have no printing.
    ignore_projections - boolean
      A flag to ignore projection differences between feature and response sets - use only if you 
      are sure the projections are really the same.

    Return: 
    features - tiles of RGB images (2*window_radius)
    responses - mask of the RGB with "f" classes in png format (2*window_radius), x is the data x dimension (2*window_radius)
          
  """
  
  if (random_seed is not None):
    np.random.seed(random_seed)

  check_data_matches(feature_file_list,response_file_list,response_vector_flag,boundary_file_list,boundary_file_vector_flag,ignore_projections)
  
  if (isinstance(samples_per_response_per_site,list)):
    if (len(samples_per_response_per_site) != len(feature_file_list)):
      raise Exception('samples_per_response_per_site must equal feature_file_list length, or be an integer.')

  if internal_window_radius is None:
    internal_window_radius = window_radius

  features = []
  name_responses = []
  responses = []

  repeat_index = []

  n_features = np.nan

  for _i in range(0,len(feature_file_list)):
    
    # open requisite datasets
    dataset = gdal.Open(feature_file_list[_i],gdal.GA_ReadOnly)
    if (np.isnan(n_features)):
      n_features = dataset.RasterCount
    feature = np.zeros((dataset.RasterYSize,dataset.RasterXSize,dataset.RasterCount))
    for n in range(0,dataset.RasterCount):
      feature[:,:,n] = dataset.GetRasterBand(n+1).ReadAsArray()


    if (response_vector_flag):
      response = rasterize_vector(response_file_list[_i],dataset.GetGeoTransform(),[feature.shape[0],feature.shape[1]])
    else:
      response = gdal.Open(response_file_list[_i]).ReadAsArray().astype(float)

    if (len(boundary_file_list) > 0):
      if (boundary_file_list[_i] is not None): #n->0
        if (boundary_file_vector_flag):
          mask = rasterize_vector(boundary_file_list[_i],dataset.GetGeoTransform(),[feature.shape[0],feature.shape[1]])
        else:
          mask = gdal.Open(boundary_file_list[_i]).ReadAsArray().astype(float)
        feature[mask == boundary_bad_value,:] = nodata_value
        response[mask == boundary_bad_value] = nodata_value



    if(verbose): print(feature.shape)
    # ensure nodata values are consistent 
    if (not dataset.GetRasterBand(1).GetNoDataValue() is None):
      feature[feature == dataset.GetRasterBand(1).GetNoDataValue()] = nodata_value
    feature[np.isnan(feature)] = nodata_value
    feature[np.isinf(feature)] = nodata_value
    response[feature[:,:,0] == nodata_value] = nodata_value
    #feature[response == nodata_value,:] = nodata_value

    # finodata_value unique response values
    un_response = np.unique(response[response != nodata_value])
    print(feature_file_list[_i],'class', un_response)
    
    if (isinstance(samples_per_response_per_site,list)):
      lsps = samples_per_response_per_site[_i]
    else:
      lsps = samples_per_response_per_site

    for ur in un_response:
      ls = np.sum(response == ur)
      lsps = min(np.sum(response == ur),lsps)
    
    # loop through each unique response 
    for ur in un_response:
      coords = np.where(response == ur)
      if(verbose): print((len(coords[0]),'response locations potentially available'))
      perm = np.random.permutation(len(coords[0]))
      coords = [coords[0][perm] ,coords[1][perm]]

      for repeat in range(0,response_repeats):
        if (center_random_offset_fraction != 0):
          coords = [coords[0] + np.random.randint(-rint(center_random_offset_fraction*window_radius),rint(center_random_offset_fraction*window_radius),len(coords[0])),coords[1] + np.random.randint(-rint(center_random_offset_fraction*window_radius),rint(center_random_offset_fraction*window_radius),len(coords[1]))]
        
        # grab the specified number (up to) of values corresponding to the response of interest
        pos_len = 0
        n=0
        while (pos_len < lsps and n < len(coords[0])):
          d = feature[coords[0][n]-window_radius:coords[0][n]+window_radius,coords[1][n]-window_radius:coords[1][n]+window_radius].copy()
          if ((np.sum(d == nodata_value) <= d.size * nodata_maximum_fraction)):
           if(d.shape[0] == window_radius*2 and d.shape[1] == window_radius*2):
            r = response[coords[0][n]-window_radius:coords[0][n]+window_radius,coords[1][n]-window_radius:coords[1][n]+window_radius].copy()
            if (np.sum(r == ur) > r.size * response_minimum_fraction):
              responses.append(r)
              #print('response', np.shape(r))
             
              features.append(d)
              name_responses.append(os.path.basename(feature_file_list[_i]).split('.')[0]+"_"+str(ur))
              repeat_index.append(repeat)
              pos_len +=1
            #else:
            #  if(verbose): print('skip from min thresh (' + str(np.sum(r ==ur)) +',' + str(r.size*response_minimum_fraction) + ')')
           else:
            if(verbose): print('skip for bad shape')
  
          n += 1
          if (n % 100 == 0 and verbose):
            print((pos_len,n,len(features)))
    
 
  # stack images up
  
  features = np.stack(features)
  name_responses = np.stack(name_responses)
  responses = np.stack(responses)
  repeat_index = np.stack(repeat_index)

  # randomly permute data to reshuffle everything
  #perm = np.random.permutation(features.shape[0])
  #features = features[perm,:]
  #responses = responses[perm,:]
  #repeat_index = repeat_index[perm]
  fold_assignments = np.zeros(responses.shape[0])

  for repeat in range(0,response_repeats):
    lfa = np.zeros(np.sum(repeat_index == repeat))
    for f in range(0,n_folds):
      lfa[rint(float(f)/float(n_folds)*len(fold_assignments)):rint(float(f+1)/float(n_folds)*len(fold_assignments))]=f
    fold_assignments[repeat_index == repeat] = lfa
  del repeat_index
    
  # reshape images for the CNN
  features = features.reshape((features.shape[0],features.shape[1],features.shape[2],n_features))
  responses = responses.reshape((responses.shape[0],responses.shape[1],responses.shape[2],1))

  print(('feature shape',features.shape))
  print(('name_responses shape',name_responses.shape))
  print(('response shape',responses.shape))
  print("========= Generating ",features.shape[0]," Tiles=========")
  for item in range(0,features.shape[0]):
    #pathmask='/results/Palms_Segment/dataset/masks/'
    pathmask='./dataset/masks/' #ALOBO 
    os.makedirs(pathmask, exist_ok=True) # Create the folder if it doesn't exist
    output_png_mask = os.path.join(pathmask, name_responses[item]+'_' + str(item) + '.png')
    cv2.imwrite(output_png_mask, responses[item]) #uses openCV to keep the values as classes 0,1,2 and 3 (no scaling)
    
    #pathfeature='/results/Palms_Segment/dataset/frames/'
    pathfeature='./dataset/frames/' #ALOBO
    os.makedirs(pathfeature, exist_ok=True) # Create the folder if it doesn't exist
    output_png_frame = os.path.join(pathfeature, name_responses[item]+'_'  + str(item) + '.png')
    
    # Extract the feature image
    feature_image = features[item, :, :, 0:3]

    # Normalize to range [0, 1]
    feature_image = (feature_image - np.min(feature_image)) / (np.max(feature_image) - np.min(feature_image))

    # Apply gamma correction for contrast adjustment (gamma > 1 darkens, gamma < 1 brightens)
    gamma = 0.5  # Example value; adjust as needed
    feature_image = np.power(feature_image, gamma)

    # Scale to [0, 255] and convert to uint8
    feature_image = (feature_image * 255).astype(np.uint8)

    imageio.imwrite(output_png_frame, feature_image)
    
    
