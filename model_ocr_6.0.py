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

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('bnw_test',
                                                 target_size = (128, 128),
                                                 batch_size = 32,classes=['Sample_1','Sample_2','Sample_3','Sample_4','Sample_5','Sample_6','Sample_7','Sample_8','Sample_9','Sample_10','Sample_11','Sample_12','Sample_13','Sample_14','Sample_15','Sample_16','Sample_17','Sample_18','Sample_19','Sample_20','Sample_21','Sample_22','Sample_23','Sample_24','Sample_25','Sample_26','Sample_27','Sample_28','Sample_29','Sample_30','Sample_31','Sample_32','Sample_33','Sample_34','Sample_35','Sample_36','Sample_37','Sample_38','Sample_39','Sample_40','Sample_41','Sample_42','Sample_43','Sample_44','Sample_45','Sample_46','Sample_47','Sample_48','Sample_49','Sample_50','Sample_51','Sample_52','Sample_53','Sample_54','Sample_55','Sample_56','Sample_57','Sample_58','Sample_59','Sample_60','Sample_61','Sample_62'])


validation_set = test_datagen.flow_from_directory('bnw_test',
                                            target_size = (128,128),
                                            batch_size = 32,classes=['Sample_1','Sample_2','Sample_3','Sample_4','Sample_5','Sample_6','Sample_7','Sample_8','Sample_9','Sample_10','Sample_11','Sample_12','Sample_13','Sample_14','Sample_15','Sample_16','Sample_17','Sample_18','Sample_19','Sample_20','Sample_21','Sample_22','Sample_23','Sample_24','Sample_25','Sample_26','Sample_27','Sample_28','Sample_29','Sample_30','Sample_31','Sample_32','Sample_33','Sample_34','Sample_35','Sample_36','Sample_37','Sample_38','Sample_39','Sample_40','Sample_41','Sample_42','Sample_43','Sample_44','Sample_45','Sample_46','Sample_47','Sample_48','Sample_49','Sample_50','Sample_51','Sample_52','Sample_53','Sample_54','Sample_55','Sample_56','Sample_57','Sample_58','Sample_59','Sample_60','Sample_61','Sample_62'])


# import matplotlib.pyplot as plt
# for batch in training_set:
#   for i in range(0,9):
#     plt.subplot(330+1+i)
#     plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
#   #images=list(images).reshape(128,128,1)
#   plt.show()
#   print(images,images.shape)
  
model_class = model.fit_generator(training_set,
                         steps_per_epoch = 32,
                         epochs = 175,
                        validation_data =validation_set,
                         validation_steps = 32,callbacks=[checkpoint])

score=model.evaluate_generator(validation_set)
print('test loss',score[0])
print('test accuracy',score[1])

# predict_datagen = ImageDataGenerator(rescale = 1./255)


# predict_set = predict_datagen.flow_from_directory('predict',target_size = (128, 128),
#                                                  batch_size = 82,
#                                                  class_mode = 'categorical')
# X,y = predict_set.next()
# arr = model.predict_classes(X)
# print(arr)

