from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import numpy as np
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing import image
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model
import timeit

import warnings

img_width, img_height = 48,48
train_data_dir = "Bmp"
validation_data_dir = "test_good"
nb_train_samples =16
nb_validation_samples = 32
batch_size = 32
epochs = 200


model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

for layer in model.layers[:5]:
    layer.trainable = False

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="tanh")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="tanh")(x)
predictions = Dense(62, activation="softmax")(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)

# compile the model 
#model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.000001, momentum=0.9), metrics=["accuracy"])

# Initiate the train and test generators with data Augumentation 
train_datagen = ImageDataGenerator(
rescale = 1./255)
# horizontal_flip = True,
# fill_mode = "nearest",
# zoom_range = 0.3,
# width_shift_range = 0.3,
# height_shift_range=0.3,
# rotation_range=30)

test_datagen = ImageDataGenerator(
rescale = 1./255)
# horizontal_flip = True,
# fill_mode = "nearest",
# zoom_range = 0.3,
# width_shift_range = 0.3,
# height_shift_range=0.3,
# rotation_range=30)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width),
class_mode = "categorical")

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


model_final.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),#variation of adam , momentum(optimizer)
              metrics=['accuracy'])

# Train the model 
model_final.fit_generator(
train_generator,
steps_per_epoch = 32,
epochs = 100,
validation_data = validation_generator,
nb_val_samples = 60,
callbacks = [checkpoint])
# datagen = ImageDataGenerator(
# rescale = 1./255,
# horizontal_flip = True,
# fill_mode = "nearest",
# zoom_range = 0.3,
# width_shift_range = 0.3,
# height_shift_range=0.3,
# rotation_range=30)
# predictor = datagen.flow_from_directory(
#         'test_good',
#         target_size=(48,48),
#         batch_size=16,
#         class_mode=None,  # only data, no labels
#         shuffle=False)  # keep data in same order as labels
# print('predictor',predictor)
filenames = validation_generator.filenames
nb_samples = len(filenames)
print('total filesamples',nb_samples)
predictions = model.predict_generator(validation_generator,nb_samples)
print(predictions,"predictions first ")

predictions = np.argmax(predictions, axis=-1) #multiple categories
print(predictions,"predictions second ")

score = model.evaluate_generator(validation_generator,nb_samples )
print('Test loss:', score[0])
print('Test accuracy:', score[1])


label_map = (train_generator.class_indices)
label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
predictions = [label_map[k] for k in predictions]
print(predictions,"predictions third")
