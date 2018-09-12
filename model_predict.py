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
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LeakyReLU
from keras import backend as K
from keras.preprocessing import image
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model
#import timeit from keras.layers.advanced_activations import LeakyReLU, PReLU
# from vis.losses import ActivationMaximization
# from vis.regularizers import TotalVariation, LPNorm
# from vis.modifiers import Jitter
# from vis.optimizer import Optimizer

# from vis.callbacks import GifGenerator
# from vis.utils.vggnet import VGG16


import numpy as np


import warnings

img_rows,img_cols=128,128
input_shape = (img_rows, img_cols, 3)


model = Sequential()

model.add(Conv2D(128, kernel_size=(3, 3), activation='tanh', input_shape=input_shape))# 16 is the filter size, input 28 *28
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(LeakyReLU(alpha=0.001))

model.add(Conv2D(256, (3, 3), activation='tanh'))# depth 16 second layer,activation tanh chla h
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(LeakyReLU(alpha=0.001))

model.add(Conv2D(256, (3, 3), activation='tanh'))# depth 16 second layer,activation tanh chla h
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(LeakyReLU(alpha=0.001))

#model.add(Dropout(0.5)) # 0.25 probability of layer we want to throw

model.add(Conv2D(1024, (3, 3), activation='tanh'))# depth 16 second layer,activation tanh chla h
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(LeakyReLU(alpha=0.001))
#model.add(Dropout(0.5)) # 0.25 probability of layer we want to throw

model.add(Conv2D(1024, (3, 3), activation='tanh'))# depth 16 second layer,activation tanh chla h
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.1))


# model.add(Conv2D(32, (3, 3), activation='tanh'))# depth 16 second layer,activation tanh chla h
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5)) # 0.25 probability of layer we want to throw


# model.add(Conv2D(32, (3, 3), activation='tanh'))# depth 16 second layer,activation tanh chla h
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5)) # 0.25 probability of layer we want to throw

model.add(Flatten())#use it before dense only

# model.add(Dense(128, activation='tanh'))
# model.add(Dropout(0.5))
model.add(Dense(1024, activation='tanh'))
model.add(Dense(512, activation='tanh'))
#model.add(Dropout(0.5))
model.add(Dense(62, activation='softmax'))
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),#variation of adam , momentum(optimizer)
              metrics=['accuracy'])
keras.models.load_model('model_ocr.h5')
model.load_weights('model_wts.h5')

predict_datagen = ImageDataGenerator(rescale = 1./255)


predict_set = predict_datagen.flow_from_directory('predict',target_size = (128, 128),
                                                 batch_size = 82,
                                                 class_mode = 'categorical')
X,y = predict_set.next()
arr = model.predict_classes(X)
print(arr)

