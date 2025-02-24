from keras.models import Model, load_model
import model 
import cv2
import numpy as np


def rgbMask(tile,w,h,b):
    img = np.zeros((h, w, b),dtype='float')
    print("tile.shape",tile.shape)
    tile=cv2.resize(tile.astype('float32'),(w,h))
    print("tile.shape",tile.shape)
    #tile=tile.argmax(2)
    tile=np.argmax(tile.squeeze(),-1)
    print("tile.shape",tile.shape)
    print("np.unique(tile)",np.unique(tile))
    
	#0-Background,1-M.Flexuosa, 2-E. precatoria, 3-O. bataua
    img[tile==0,0] = 0 #Empty class 
    img[tile>=1,0] = 50 #Clase aguaje
    img[tile==2,0] = 100 #Clase -E. precatoria
    img[tile==3,0] = 150 #Clase O. bataua
    return img
#load the model to test prediction
test_model = load_model('models/deeplab_keras_model_palms_test3_RESCALADO-AUG_0.0001.h5',custom_objects={'relu6':model.relu6,'BilinearUpsampling':model.BilinearUpsampling,'dice_coef':model.dice_coef })
#test_model = load_model('models/deeplab_keras_model_palms_test3_RESCALADO-SINAUG_0.003.h5',custom_objects={'relu6':model.relu6,'BilinearUpsampling':model.BilinearUpsampling})

name_file='VEN-01_03_4_2.0_2512.png'
tempBlock=np.zeros((1,512,512,3)).astype(np.float)
train_img = cv2.imread('/mnt/guanabana/raid/home/xtagle/ML/CNN/ecoCNN/dataset/frames/'+name_file) #/DMM-02_1_1.0_1943.png
train_img=train_img / 127.5 - 1 
tempBlock[0,] = train_img  
a = test_model.predict(tempBlock)
print("a[0].shape",a[0].shape)
print("np.unique(a[0][0])",np.unique(a[0][0]))
print("np.unique(a[0][1])",np.unique(a[0][1]))

#a = np.argmax(a.squeeze(),-1)
#print(np.unique(a))
#plt.imsave('002_53.jpg', a, cmap=cm.gray)
img=rgbMask(a[0],512,512,1)
cv2.imwrite('pred/test_predic_0001_aug-'+name_file, img)
