import fnmatch, os
import numpy as np
import cv2

train_frame_path = './dataset/frames/'
train_mask_path = './dataset/masks/'
path="./dataset/"
n = fnmatch.filter(os.listdir(train_frame_path), "*.png") # include dot-files
print('Images: ',len(n))

def split_train_val(list_tiles):
    ### Split dataset (80/20)
    TRAINING_SPLIT = 0.8 #80%
    VAL_SPLIT=0.2
    TEST_SPLIT=0.0  # Independent dataset

    it = int(len(list_tiles) * TRAINING_SPLIT)
    iv = int(len(list_tiles) * VAL_SPLIT)
    itt = int(len(list_tiles) * TEST_SPLIT)
    print('it:',it,' - iv',iv,' itt:',itt)

    #n = np.random.permutation(n)

    listtrain=list_tiles[0:it]
    listval=list_tiles[it:]
    return listtrain,listval


def generarlistfromfilter(listmask):
    list_tiles1=[]
    list_tiles2=[]
    list_tiles3=[]
    for mask in listmask:
        imgmask = cv2.imread(train_mask_path+'/'+mask, cv2.IMREAD_GRAYSCALE)#read grey scale (1D)
        uqclases=np.unique(imgmask)

        if 3 in uqclases:
            list_tiles3.append(mask)
        elif 2 in uqclases:
            list_tiles2.append(mask)
        elif 1 in uqclases:
            list_tiles1.append(mask)
    min_size = min(len(list_tiles1), len(list_tiles2), len(list_tiles3))
    #min_size=6000
    print("class 1",len(list_tiles1))
    print("class 2",len(list_tiles2))
    print("class 3",len(list_tiles3))
    print("min_size",min_size)
    np.random.shuffle(list_tiles1)
    np.random.shuffle(list_tiles2)
    np.random.shuffle(list_tiles3)
    list_tiles1=list_tiles1[0:min_size]
    list_tiles2=list_tiles2[0:min_size]
    list_tiles3=list_tiles3[0:min_size]

    listtrain1,listval1=split_train_val(list_tiles1)
    listtrain2,listval2=split_train_val(list_tiles2)
    listtrain3,listval3=split_train_val(list_tiles3)
    
    b_listtrain = listtrain1+listtrain2+listtrain3
    b_listval= listval1+listval2+listval3
    np.random.shuffle(b_listtrain)
    np.random.shuffle(b_listval)
    return b_listtrain,b_listval

#Reduce training data size for faster demo training
n=n[0:50]
listtrain,listval = generarlistfromfilter(n)

print("listtrain",len(listtrain))
print("listval",len(listval))

###Save as a text file
np.savetxt(path+'trainlist.txt', listtrain, delimiter='\\n',fmt='%s')
np.savetxt(path+'vallist.txt', listval, delimiter='\\n',fmt='%s')

