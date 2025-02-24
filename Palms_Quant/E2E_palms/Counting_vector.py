#!/usr/bin/env python
# coding: utf-8
##### Script to polygonize the output from DWT ####

### Import libraries needed
from osgeo import gdal, ogr, osr

import geopandas as gd
import pandas as pd
import os


### Select the nodes that will be used
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
##Guanabana
#os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3" #0-3
##grenadilla
os.environ["CUDA_VISIBLE_DEVICES"]="2,3" #0-4

###Raster Input
#If only one raster
#in_path = '/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/VAP-01_1_2__dwt.tif'

#dl_bdwt

list_in_path=[
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/R1/VEN-01_01_02__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/R2/NAR-01_1_2_3_4_5__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/R2/DMM-02_1__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/R3/VEN-02_01_2__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/R3/PRN-01_01__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/R4/AGU-01__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/R4/CHE-03_3__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/R5/PRN-01_11_12_13__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/R5/VAP-01_1_2__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/R6/NJN-01_4_5__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/R6/NJN-01_6__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/Clip_AM-02__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/VEA-01_1_transparent_mosaic_group1__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/CIE-01_1__dwt.tif',
'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/SNB-01__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/SJR-01_6__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/CIJH_VIV_1__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/JEN-11_1_2_3_4_5__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/VAP-03_1_2_3_4__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/QUI-01_14__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/PISC-01_5__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/NJN-01_6__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/CHE-04_1_2_3__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/GAL-01_1__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/POL-01_1__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/SJR-01_1__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/KIN-01__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/PISC-02_9__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/VEN-04_06_7_8__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/AMA-01__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/VEN-05_08_9_10__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/VEN-03_9_10_11__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/JEN-15_10_11__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/CHE-03_3__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/VEN-02_8_9_10__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/VEN-01_09_10_11__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/VEN_total_1__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/VEN_total_2__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/PIU-03_1_5__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/HUA-01_1_2_3_4_5__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/SAM-01_13__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/SJO-00_01_2__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/MSH-01_1__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/CHE-05_1__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/SJO-01_01_2_3__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/NYO-00_1__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/PRN01_11_12_13__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/NAR-01_1_2_3_4_5_all__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/Clip_Brigida220622_1__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/Clip_Elina210622_1__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/Clip_Sandoval31_07_1__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/Sandoval_Aguajal__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/NYO-03_1_5_6_7_8_9_transparent_mosaic_group1__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/PIU-00_1_2_3_4_5_transparent_mosaic_group1__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/NJN-01_4_5_all__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/dl_bdwt/NJN-01_4_5_all__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/VAP-02_1__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/AGU-01__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/Clip_ALP-60_8_9_10_11_12_13_14_21_22__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/Clip_ALP-60_8_9_10_11_12_13_14_21_22_TEST__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/Brigida220622_1__dwtT.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/Brigida220622_2__dwtT.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/Clip_PRN-01_11_12_13__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/Clip_PRN-01_11_12_13_TEST__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/DMM-02_1__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/Elina210622_1__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/Elina210622_2__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/NAR-01_1_2_3_4_5_all__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/NJN-01_4_5_all__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/JEN-14_04_5_6_7__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/JHU-01_9_10_11_12__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/VEN-00_1_2_3__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/PIU-02_9_13_14_15_16_17_transparent_mosaic_group1__dwt.tif'
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/PISC-01_1__dwt.tif' 
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/Clip_PIU-02_9_13_14_15_16_17_TEST__dwt.tif', 
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/PIU-02_9_13_14_15_16_17_transparent_mosaic_group1__dwt.tif', 
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/VAP-01_1_2__dwt.tif',
#'/mnt/grenadilla/raid/home/xtagle/ML/CNN/dwt/output/e2e/raster/test/All/Clip_VAP-01_1_2_TEST__dwt.tif',
]

for in_path in list_in_path:
    if  os.path.exists(in_path)==False:
        print("XXXXXXXXX NOT FOUND xxxxxxxxxxX")
        print("in_path",in_path)
        print("XXXXXXXXX END xxxxxxxxxxX")
        continue
    print("=====================START================")
    print("in_path",in_path)
    ###Output info
    out_POL_path = os.path.join(in_path.split('.tif')[0] + '_poly.gpkg')
    out_CEN_path = os.path.join(in_path.split('.tif')[0] + '_centers.gpkg')
    out_CSV_path = os.path.join(in_path.split('.tif')[0] + '_atributos.csv')
    out_REPORT_path = os.path.join(in_path.split('.tif')[0] + '_report.csv')
    print(out_POL_path)
    ###GDAL polygonize
    #  get raster datasource
    src_ds = gdal.Open( in_path )
    #
    srcband = src_ds.GetRasterBand(1)
    dst_layername = 'palms_Area_ha'
    drv = ogr.GetDriverByName("GPKG")
    #drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource( out_POL_path )
    prj=src_ds.GetProjection()

    sp_ref = osr.SpatialReference()
    sp_ref = osr.SpatialReference(wkt=prj)
    #sp_ref.SetFromUserInput('EPSG:4326')

    dst_layer = dst_ds.CreateLayer(dst_layername, srs = sp_ref )

    fld = ogr.FieldDefn("ID", ogr.OFTInteger)
    dst_layer.CreateField(fld)
    dst_field = dst_layer.GetLayerDefn().GetFieldIndex("ID")


    gdal.Polygonize( srcband, None, dst_layer, dst_field, [], callback=None )

    del src_ds
    del dst_ds

    ###Get Centroids

    gdf = gd.read_file(out_POL_path)
    gdf.info()
    gdf=gdf[gdf["ID"]>0]


    ###Function for the table

    c1=0 #count each palm
    c2=0
    c3=0
    ca1=0 #sum all the area
    ca2=0
    ca3=0
    def generateCentroidColumns(item):
        global c1
        global c2
        global c3
        global ca1
        global ca2
        global ca3
        ESPECIE = 'Mauritia flexuosa'
        geom = item.geometry
        if (item['ID'] == 15) : #0=FIELD ID
            ESPECIE = 'Mauritia flexuosa' #2= FIELD ESPECIE
            c1 = c1+1
            ca1 = ca1 + geom.area
        elif (item['ID'] == 25):
            ESPECIE = 'Euterpe precatoria'
            c2 = c2+1
            ca2 = ca2 + geom.area
        elif(item['ID'] == 35):
            ESPECIE = 'Oenocarpus bataua'
            c3 = c3+1
            ca3 = ca3 + geom.area

        areacopa=geom.area  #3= FIELD ESPECIE
        cx = geom.centroid.x#4= FIELD UTMESTE
        cy = geom.centroid.y #5= FIELD UTMNORTE
        clase = item['ID'] #1= CLASE
    

        return pd.Series([clase, ESPECIE, areacopa,cx,cy])


        
    ###Apply function to obtain centroids with info
    gdf[['ID','ESPECIE','ÁREA(m2)','UTM(ESTE)','UTM(NORTE)']]=gdf.apply(generateCentroidColumns,axis=1)

    print(gdf.head())

    def generarcentroid(item):
        return pd.Series([item.ESPECIE, item.geometry.centroid])

    dfcentroid=gdf.apply(generarcentroid,axis=1).set_axis(['ESPECIE', 'geometry'], axis=1, inplace=False)
    dfcentroid.set_geometry(col='geometry', inplace=True)
    print(dfcentroid.head())
    gdf.to_file(out_POL_path)
    dfcentroid.to_file(out_CEN_path)

    gdf.to_csv(out_CSV_path, sep =',')
    labelnames = ['ESPECIE', 'CANTIDAD DE INDIVIDUOS', 'AREA TOTAL(ha)' ]
    rowname1 = ['Mauritia flexuosa', c1, ca1/10000 ]
    rowname2 = ['Euterpe precautoria', c2, ca2/10000]
    rowname3 = ['Oenocarpus bataua', c3, ca3/10000]
    with open(out_REPORT_path, 'w') as output_file:
        line = ','.join(name for name in labelnames) + '\n'
        output_file.write(line)
        line = ','.join(str(name) for name in rowname1) + '\n'        
        output_file.write(line)
        line = ','.join(str(name)for name in rowname2) + '\n'        
        output_file.write(line)
        line = ','.join(str(name) for name in rowname3) + '\n'        
        output_file.write(line)

    print("=====================END================")


	
	
