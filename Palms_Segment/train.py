import numpy as np
from model import Deeplabv3
from util import *
from keras import optimizers 
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

import tensorflow as tf
from keras.optimizers import Adam


NO_OF_EPOCHS = 200

BATCH_SIZE = 8

# dimensiones de las imagenes
img_width, img_height = 512, 512


frame_path = 'results/Palms_Segment/dataset/frames'
mask_path = 'results/Palms_Segment/dataset/dataset/masks'

train_list = 'results/Palms_Segment/dataset/dataset/trainlist.txt'
val_list = 'results/Palms_Segment/dataset/dataset/vallist.txt'

NO_OF_TRAINING_IMAGES = len(np.genfromtxt(train_list, delimiter="\\n",dtype='str'))
NO_OF_VAL_IMAGES = len(np.genfromtxt(val_list, delimiter="\\n",dtype='str'))


# Train the model
train_gen = generar_datos(train_frame_path,train_mask_path,train_list, batch_size = BATCH_SIZE)
val_gen = generar_datos(val_frame_path,val_mask_path,val_list, batch_size = BATCH_SIZE)

deeplab_model = Deeplabv3(input_shape=(img_width, img_height, 3), classes=4, OS = 16)


deeplab_model.compile(optimizer=Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss="categorical_crossentropy", metrics=["categorical_accuracy"])
callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('deeplab_keras_model.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

results = deeplab_model.fit_generator(train_gen, epochs=NO_OF_EPOCHS, 
                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                          validation_data=val_gen, 
                          validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE),
                          callbacks=callbacks)

deeplab_model.save_weights('deeplab_keras_weights.h5')
deeplab_model.save('deeplab_keras_model.h5')