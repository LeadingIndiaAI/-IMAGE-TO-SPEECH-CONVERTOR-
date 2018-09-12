from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import numpy as np
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing import image
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model
import timeit

import warnings

img_rows,img_cols=64,64
input_shape = (img_rows, img_cols, 3)

model = Sequential()
model.add(Conv2D(16, kernel_size=(5, 5), activation='tanh', input_shape=input_shape))# 16 is the filter size, input 28 *28
model.add(Conv2D(32, (5, 5), activation='tanh'))# depth 16 second layer,activation tanh chla h
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5)) # 0.25 probability of layer we want to throw
model.add(Conv2D(32, (5, 5), activation='tanh'))# depth 16 second layer,activation tanh chla h
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5)) # 0.25 probability of layer we want to throw
model.add(Conv2D(32, (5, 5), activation='tanh'))# depth 16 second layer,activation tanh chla h
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5)) # 0.25 probability of layer we want to throw
model.add(Conv2D(32, (5, 5), activation='tanh'))# depth 16 second layer,activation tanh chla h
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5)) # 0.25 probability of layer we want to throw

model.add(Flatten())
model.add(Dense(32, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(62, activation='softmax'))
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),#variation of adam , momentum(optimizer)
              metrics=['accuracy'])
model.save_weights('model_wts.h5')

# Save Model
model.save('model_ocr.h5')
checkpoint=ModelCheckpoint('model_ocr.h5',
                            monitor='val_acc',
                            verbose= 1,
                            save_best_only=True,
                            save_weights_only=False,
                            mode='auto',
                            period=1)
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('Bmp',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('test_good',
                                            target_size = (64,64),
                                            batch_size = 32,
                                            class_mode = 'categorical')
model_class = model.fit_generator(training_set,
                         steps_per_epoch = 32,
                         epochs = 100,
                        validation_data = test_set,
                         validation_steps = 60,callbacks=[checkpoint])