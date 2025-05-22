import sys,os
import re
from osgeo import gdal, ogr
import rasterio.features
import numpy as np
import matplotlib.pyplot as plt
import fiona

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

### Apply function

### Load data
#UAV mosaics
feature_file_list = ['../data/DMM-02_1.tif']
dataset = gdal.Open(feature_file_list[0],gdal.GA_ReadOnly)#Reading all the images --> dataset
print(dataset)

#Crowns shapefiles
response_file_list = ['../data/Mflexuosa_DMM02_2019.shp',
                      '../data/Eprecatoria_DMM02_2019.shp',
                    '../data/Obataua_DMM02_2019.shp'
                     ]

### Output folder
out_folder = '../results/Palms_Segment/responses_raster' #do not forget to create the folder if it does not exist
os.makedirs(out_folder, exist_ok=True) # Create the folder if it doesn't exist

#Run the function
response = np.zeros((dataset.RasterYSize,dataset.RasterXSize))

for i in range(0,len(response_file_list)):
    aux = rasterize_vector(response_file_list[i],dataset.GetGeoTransform(),[dataset.RasterYSize,dataset.RasterXSize])#rasterize the response
    response = response + aux*(i+1)

output_tif_file = os.path.join(out_folder, 'Palms_merged_'+ os.path.basename(feature_file_list[0]).split('.')[0]  + '_classes.tif')

driver = gdal.GetDriverByName('GTiff')
outrgb = driver.Create(output_tif_file,dataset.RasterXSize,dataset.RasterYSize,1,gdal.GDT_Float32)
outrgb.SetProjection(dataset.GetProjection())
outrgb.SetGeoTransform(dataset.GetGeoTransform())
outrgb.GetRasterBand(1).WriteArray(response[:,:],0,0)
del outrgb
print ("***********Rasters saved, check folder", out_folder, "********************")
