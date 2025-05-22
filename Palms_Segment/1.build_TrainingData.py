##### Script to generate the training data with 4 classes ####

"""
Script to generate the training data to Train a Deeplabv3 Model for Palm Species Detection

This script is designed to generate training data to train a model to detect three palm species in Amazonian forests
using drone-based imagery and the Deeplabv3 architecture.

Classes: 4

Background (class 0)
Mauritia Flexuosa (class 1)
Euterpe Precatoria (class 2)
Oenocarpus Bataua (class 3)



This code belongs to:
Tagle et al. (2024).'Overcoming the Research-Implementation Gap through Drone-based Mapping of Economically Important Amazonian Palms'

The code was adapted from:
- https://github.com/pgbrodrick/ecoCNN


Requirements:
- Python 3.x
- OpenCV


Authors: Tagle,X.; Cardenas, R.; Palacios, S.; Marcos, D.
"""

### Import libraries needed
import os
#Import our modules
#from Palms_Segment import generate_training_data
import generate_training_data

### Select the nodes that will be used
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


### Prepare training data: Samples+Responses+ ROIs

# Ignore the projection in case of GDAL errors
ignore_projections=True #False

# Windows size selection: number of pixels along one side of an image square, higher the window, more context, more training time
window_radius = 256 #256 for the palm trees, higher takes longer
samples_per_response_per_site = 191 #30 #100 #170 for only Mauritia

# List of mosaics 
feature_file_list = [ '../data/DMM-02_1.tif'
#'../data/DMM-01_10.tif'
# 'data/AGU-01_crop.tif',
#                     'data/PRN-01_01.tif',
#   './data/AM_02.tif',
#'./data/DMM-01_10.tif',
#'./data/FDJH-01_1_2.tif', 
	]

# Rasters with the response data (the function supports shapefiles directly but only when working with one class)
response_file_list = ['../results/Palms_Segment/responses_raster/Palms_merged_DMM-02_1_classes.tif'
#'../responses_raster/Palms_DMM-02_classes.tif'
#'responses_raster/Palms_AGU-01_crop_classes.tif',
#                      'responses_raster/Palms_PRN-01_01_classes.tif',
#'./responses_raster/Palms_merged_AM_02_classes.tif',
#'./responses_raster/Palms_merged_DMM-01_10_classes.tif',
#'./responses_raster/Palms_merged_FDJH-01_1_2_classes.tif',
	] 

# Shapefile with the Area of Interest (ROI)
boundary_file_list = ['../data/DMM02_ROI_2019.shp'
#'../data/DMM02_ROI_2019.shp'
#'data/AGU01_ROI_2018.shp',
#                      'data/PRN01_ROI_2017.shp',
#'./data/AM02_ROI_2019.shp',
#'./data/DMM01_ROI_2019.shp',
#'./data/FDJH01_ROI_2019.shp',
	]


# Offset the window to relocate the response -> not having the response centered, increase overall sample size 
internal_window_radius = int(round(window_radius*0.75))



#Prepare training data: 4dimensional data arrays with dimensions (number of samples, y, x, number of features) and (number of samples, y, x, number of responses)
generate_training_data.build_semantic_segmentation_training_data(
                                      window_radius,
                                      samples_per_response_per_site,
                                      feature_file_list,
                                      response_file_list,
									  response_vector_flag=False, #True when working with vectors (only one class)
									  boundary_file_list=boundary_file_list,
									  boundary_file_vector_flag=True, #True when using ROIs
									  internal_window_radius=internal_window_radius,
									  fill_in_feature_data=True,
									  ignore_projections=ignore_projections,
									  #global_scale_flag='mean_std', #Scaling, Default None
									  local_scale_flag='mean', #scaling
                                      nodata_maximum_fraction=0.8,
                                      center_random_offset_fraction=0.0, 
                                      response_minimum_fraction=0.0,#0.5
									  response_repeats=1, # To sample from each response center n times
									  random_seed=13, #for reproducibility
									  n_folds=10,
									  verbose=True, #False to have no output printing 
									  savename='test') #Default None

print("======>Finished")
