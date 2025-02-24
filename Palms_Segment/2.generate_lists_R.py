##### Script to list the images for training and validation of the Palm Species Mapping####


### Import libraries needed
import fnmatch, os
import numpy as np
import pandas as pd
import re

### Select the nodes that will be used
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

### Split dataset
def Filter(datalist,caracter):
    # Search data based on regular expression in the list
    return [val for val in datalist
        if re.search(r'^'+caracter, val)]

dfrasters=pd.read_csv('Lista_raster.csv', sep=';')
Replicat=["R1","R2","R3","R4","R5","R6"]
train_frame_path = 'frames/'
train_mask_path = 'masks/'
path=""
list_tiles = fnmatch.filter(os.listdir(train_frame_path), "*.png") # include dot-files
print('Tiles: ',len(list_tiles))


TRAINING_SPLIT = 0.80
VAL_SPLIT=0.20
TEST_SPLIT=0.0 # Independent UAV mosaics

for x in range(0,len(Replicat)):
	list_m_tiles=[]
	df_rm=dfrasters[dfrasters['Replicates'].str.contains(Replicat[x])]
	print("====>Replicate ",Replicat[x],"=>",len(df_rm)," raster")
	for index, row in df_rm.iterrows():
		tileds_f_raster=Filter(list_tiles,row['Name_raster'])
		print(row['Name_raster'],"=>",len(tileds_f_raster)," tiles")
		list_m_tiles=np.append(list_m_tiles, tileds_f_raster)

	print("====>Replicate ",Replicat[x],"=>",len(list_m_tiles)," tiles")
	print("\n")
	it = int(len(list_m_tiles) * TRAINING_SPLIT)
	iv = int(len(list_m_tiles) * VAL_SPLIT)
	itt = int(len(list_m_tiles) * TEST_SPLIT)
	print('it:',it,' - iv',iv,' itt:',itt)
	list_m_tiles = np.random.permutation(list_m_tiles)
	listtrain=list_m_tiles[0:it]
	listval=list_m_tiles[it:]
	#listtest=n[it+iv:]
	np.savetxt(path+Replicat[x]+'_trainlist.txt', listtrain, delimiter='\\n',fmt='%s')
	np.savetxt(path+Replicat[x]+'_vallist.txt', listval, delimiter='\\n',fmt='%s')
	#np.savetxt(path+'testlist.txt', listtest, delimiter='\\n',fmt='%s')
	





