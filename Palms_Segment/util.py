##### Functions used during the training ####
import numpy as np
import random
import os
import tensorflow.keras.backend as K
import cv2

def iou_coef(y_true, y_pred, smooth=1):
    """
    IoU = (|X & Y|)/ (|X or Y|)
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection

    return (intersection + smooth) / (union + smooth)


def iou_coef_metric(y_true, y_pred):
    return iou_coef(y_true, y_pred)

def generar_datos(img_folder, mask_folder, batch_size):
  c = 0
  n = os.listdir(img_folder) #List of training images
  random.shuffle(n)
  
  while (True):
    img = np.zeros((batch_size, 512, 512, 3)).astype('float')
    mask = np.zeros((batch_size, 512, 512, 4)).astype('int8')

    for i in range(c, c+batch_size): #initially from 0 to 16, c = 0. 

      train_img = cv2.imread(img_folder+'/'+n[i]) 

      imask = np.zeros((512, 512, 4)).astype('int8')
      train_mask = cv2.imread(mask_folder+'/'+n[i], cv2.IMREAD_GRAYSCALE)#escala de grises una sola dimension
      
      imask[train_mask==0, 0]= 1 # [1,0,0,0] Clase vacia
      imask[train_mask==1, 1]= 1 #[0,1,0,0] Clase aguajes
      imask[train_mask==2, 2]= 1 #[0,0,1,0] Clase copas
      imask[train_mask==3, 3]= 1 #[0,0,0,1] Clase copas
      
      
      img[i-c] = train_img     

      mask[i-c] = imask

    c+=batch_size
    if(c+batch_size>=len(os.listdir(img_folder))):
      c=0
      random.shuffle(n)
                  # print "randomizing again"
    yield(img,mask)