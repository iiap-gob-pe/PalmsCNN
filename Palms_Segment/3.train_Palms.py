##### Script to train model to detect 3 palm species using Deeplabv3 ####
"""
Script to Train Deeplabv3 Model for Palm Species Detection

This script is designed to train a model to detect three palm species in Amazonian forests
using drone-based imagery and the Deeplabv3 architecture. 

This code belongs to:
Tagle et al. (2024).'Overcoming the Research-Implementation Gap through Drone-based Mapping of Economically Important Amazonian Palms'

The code leverages architectures and techniques from the repositories:
- https://github.com/pgbrodrick/ecoCNN
- https://github.com/bonlime/keras-deeplab-v3-plus

Requirements:
- Python 3.x
- TensorFlow/Keras
- OpenCV
- imgaug

Authors: Tagle,X.; Cardenas, R.; Palacios, S.; Marcos, D.
"""

### Import necessary libraries
import numpy as np
from model import Deeplabv3
from util import *
#from tensorflow.keras #ALOBO
#from keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
#from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

np.bool=bool
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import os

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils import losses_utils
from model import Deeplabv3, dice_coef

import datetime
import cv2
from math import sqrt
import imageio
# import random  # Uncomment if needed

### Configure GPU settings and select the nodes to be used in the cluster
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Uncomment to use CPU instead of GPU

### Parameter Initialization
TRAINING_SPLIT = 0.8  # Percentage of data used for training
NO_OF_EPOCHS = 15     # Number of epochs for training
BATCH_SIZE = 8        # Batch size for training
SEED = 13             # Seed for reproducibility
LR = 0.003 #Learning rate 0.005 0.01 0.013

#Image dimensions (Number of pixels)
img_width, img_height = 512, 512
NUM_CHANNELS = 3 #RGB
rep="all" #R1, R2,R3,R4,R5,R6, all
typetest="iaa"

### Split data in Training and Validation
#Paths
train_frame_path = 'dataset/frames' #X  #ALOBO
train_mask_path = 'dataset/masks'   #Y  #ALOBO 
train_list = 'dataset/'+rep+'_trainlist.txt'
val_list = 'dataset/'+rep+'_vallist.txt'


log_path = "./logs/fit/" #ALOBO

#Print info 
NO_OF_IMAGES = len(os.listdir(train_frame_path))
NO_OF_TRAINING_IMAGES = len(np.genfromtxt(train_list, delimiter="\\n",dtype='str'))
NO_OF_VAL_IMAGES = len(np.genfromtxt(val_list, delimiter="\\n",dtype='str'))
#NO_OF_VAL_IMAGES = len(os.listdir(val_frame_path))
print("NO_OF_TRAINING_IMAGES : ",NO_OF_TRAINING_IMAGES)
print("NO_OF_VAL_IMAGES : ",NO_OF_VAL_IMAGES)
#4 Classes: 0-Background,1-M.Flexuosa, 2-E. precatoria, 3-O. bataua

#Class Weights
def create_class_weight(n_total_samples,n_class_samples):
	w = sqrt(1/((n_total_samples // n_class_samples)+1e-6))
	return w

def getWeightedClass(mask_folder,list_mask):
	n = np.genfromtxt(list_mask, delimiter="\\n",dtype='str')
	samples_class={0:0, 1:0, 2:0, 3:0} #0:0
	for i in range(0, len(n)):
		mask = cv2.imread(mask_folder+'/'+n[i], cv2.IMREAD_GRAYSCALE)#read grey scale (1D)
		uqclases=np.unique(mask)
		if 0 in uqclases:
			samples_class[0]+=1
		if 1 in uqclases:
			samples_class[1]+=1
		if 2 in uqclases:
			samples_class[2]+=1
		if 3 in uqclases:
			samples_class[3]+=1
	print("samples_class",samples_class)		
	return  [create_class_weight(len(n),samples_class[0]),
			 create_class_weight(len(n),samples_class[1]),
			 create_class_weight(len(n),samples_class[2]),
			 create_class_weight(len(n),samples_class[3])
			]



#class_weight=getWeightedClass(train_mask_path,train_list)
#print("class_weight",class_weight)	

"""
class WeightedCategoricalCrossentropy(keras.losses.CategoricalCrossentropy):
    def _init_(
        self,
        weights,
        from_logits=False,
        label_smoothing=0,
        reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
        name='categorical_crossentropy',
    ):
        super()._init_(
            from_logits, label_smoothing, reduction, name=f"weighted_{name}"
        )
        self.weights = weights
    def call(self, y_true, y_pred):
        weights = self.weights
        nb_cl = len(weights)
        final_mask = keras.backend.zeros_like(y_pred[:, 0])
        y_pred_max = keras.backend.max(y_pred, axis=1)
        y_pred_max = keras.backend.reshape(
            y_pred_max, (keras.backend.shape(y_pred)[0], 1))
        y_pred_max_mat = keras.backend.cast(
            keras.backend.equal(y_pred, y_pred_max), keras.backend.floatx())
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
            final_mask += (
                weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
        return super().call(y_true, y_pred) * final_mask
		
	
NUM_KEYPOINTS = 2
	
		
def generator(frames, masks, batch_size):
	return_features = frames.copy()
	return_labels = masks.copy()
	# create empty arrays to contains batch of features and labels
	batch_frames = np.zeros((BATCH_SIZE, img_width, img_height, NUM_CHANNELS))
	batch_masks = np.zeros((BATCH_SIZE, 2 * NUM_KEYPOINTS))	# X and Y coordinates
	while True:
		for i in range(batch_size):
			
			# choose random index in features
			index = randint(0, len(return_features)-1)
			
			random_augmented_image, random_augmented_labels = augment(np.array([features[index]]), np.array([labels[index]]))
			batch_frames[i] = random_augmented_image[0]
			batch_masks[i] = random_augmented_labels[0]
		yield batch_frames, batch_masks
"""


#Function to export augmented images and masks
def save_iaa_img(path,namefile,frame,mask):
	imask = np.zeros((512, 512, 1)).astype('int8')
	imask[mask[:,:,1]==1]=50
	imask[mask[:,:,2]==2]=100
	imask[mask[:,:,3]==3]=150
	#namefile=namefile.split(".")[0]
	output_png_mask = os.path.join(path, namefile+'_mask'+ '.png')
	cv2.imwrite(output_png_mask, imask) #uses openCV to keep the values as classes 0,1,2 and 3 (no scaling)
	output_png_frame = os.path.join(path, namefile)
	imageio.imwrite(output_png_frame, frame)

def data_generator(img_folder, mask_folder, list_img,batch_size,secaug=False,isdatatrain=False):  #secaug=False without iaa 
  #http://www.jessicayung.com/using-generators-in-python-to-train-machine-learning-models/
  samples = np.genfromtxt(list_img, delimiter="\\n",dtype='str')
  #n = os.listdir(img_folder) #List of training images
  num_samples = len(samples)
  if isdatatrain:
  	print("num_samples: ",num_samples)
  while (True):
    random.shuffle(samples)
    
    # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size &lt;= num_samples]
    for offset in range(0, num_samples, batch_size):
    	# Get the samples you'll use in this batch
        if isdatatrain:
        	print("offset",offset)
        
        batch_samples = samples[offset:offset+batch_size]

        # Initialize X_train and y_train arrays for this batch
        X_train = []
        y_train = []
        for batch_sample in batch_samples:
	        img_file = cv2.imread(img_folder+'/'+batch_sample)	              
	        mask_file = cv2.imread(mask_folder+'/'+batch_sample, cv2.IMREAD_GRAYSCALE)#1 dim grey scale
	        #print("mask_file",mask_file)
	        train_img=img_file
	        train_mask=mask_file
	        imask = np.zeros((512, 512, 4)).astype('int8')
	        imask[train_mask==0, 0]= 1 # [1,0,0,0] Background class
	        imask[train_mask==1, 1]= 1 #[0,1,0,0] Mauritia class
	        imask[train_mask==2, 2]= 1 #[0,0,1,0] Euterpe class
	        imask[train_mask==3, 3]= 1 #[0,0,0,1] Oenocarpus class
		        
	        X_train.append(train_img)
	        y_train.append(imask)

		
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        if secaug:
        	images_aug, segmaps_aug = secaug(images=X_train, segmentation_maps=y_train)
        	#for item in range(0, batch_size):
        	#	save_iaa_img(path="data_iaa",namefile="frame_test"+str(item)+".png",frame=images_aug[item],mask=segmaps_aug[item])

        	#scale data to the range of [0, 1]
        	#images_aug=images_aug.astype("float32") / 255.0
			#scale data to the range of [-1, 1] 
        	images_aug=images_aug / 127.5 - 1 
        	yield(images_aug,segmaps_aug)
        else:
        	# scale data to the range of [0, 1]
        	#X_train=X_train.astype("float32") / 255.0
			# scale data to the range of [-1, 1]
        	X_train=X_train.astype("float32")/ 127.5 - 1 
        	yield(X_train,y_train)

###Image augmentation
aug = iaa.SomeOf((0, 2), [ # Perform 0 to 2 of the following augmenters
	iaa.MultiplyBrightness((0.8, 1.2)), #change brightness -20 - 20% of the img resembling ilumination conditions
	iaa.MultiplySaturation((0.8, 1.1)), #change saturation 10-150% of the img
	iaa.Fliplr(0.5),		# Flip horizontally 50% of the img
	iaa.Flipud(0.5),		# Flip vertically 50% of the img
	iaa.Affine(rotate=(-20, 20)), # rotate by -20 to +20 degrees
	iaa.Affine(scale={"x": (1, 1), "y": (1, 1)}), # zoom in and out by 80-120%0.8 1.2
	iaa.imgcorruptlike.Fog(severity=1),	# severity: 1-5 Add blur resembling fog/water droplets
	iaa.Multiply((0.8, 1.3)), # makes images brighter or darker
	iaa.ElasticTransformation(alpha=(0, 2), sigma=0.9), #Alpha 0-5, sigma 0.01-1 resembling artifacts in the mosaics
	iaa.JpegCompression(compression=(60, 70)), #Degrade the quality of images by JPEG-compressing them, 0-100 (100 higher compression)-> different sensors
	iaa.MotionBlur(k=(4,5)), #Blur images in a way that fakes camera or object movements 5x5 to 11x11 kernels -> wind or UAV movement
	])

### Model settings
deeplab_model = Deeplabv3(input_shape=(img_width, img_height, 3), classes=4, OS = 16)
#deeplab_model.load_weights('models/deeplab_keras_weights_palms_test1_20220622_0.003.h5', by_name=True) #Fine tuning: deeplab_keras_model_palms or deeplab_keras_model_palms_all.h5 
deeplab_model.compile(optimizer=Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss="categorical_crossentropy", metrics=["categorical_accuracy", dice_coef]) #lr=0.003

log_dir = log_path + rep + '_'+typetest+'_'+ str(LR) +'_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0,write_graph = True)

namemodel='models/deeplab_keras_model_palms_'+rep+'_'+typetest+'_'+str(LR)+ '.h5'
callbacks = [
    #EarlyStopping(patience=10, verbose=1),
    #ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint(namemodel, verbose=1, save_best_only=True, save_weights_only=False), 
	tensorboard_callback
	]

# Train the model
train_gen = data_generator(train_frame_path,train_mask_path,train_list, batch_size = BATCH_SIZE,secaug=aug, isdatatrain=True)# secaug=aug isdatatrain=False
val_gen = data_generator(train_frame_path,train_mask_path,val_list, batch_size = BATCH_SIZE,secaug=False,isdatatrain=False)

results = deeplab_model.fit_generator(train_gen,
                          epochs=NO_OF_EPOCHS, 
                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                          validation_data=val_gen, 
                          validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE),
						  #class_weight=class_weight
                          callbacks= callbacks
                          )
print("Name of model: ", namemodel)
# Save model
#deeplab_model.save_weights(namemodel) 
#deeplab_model.save('models/deeplab_keras_model_palms_'+rep+'_'+typetest+'_20220623_2_'+str(LR)+ '.h5') #iaa_


##############
