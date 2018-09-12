from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import numpy as np
from keras.preprocessing import image
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model
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

img_width, img_height = 48,48
train_data_dir = "Bmp"
validation_data_dir = "test_good"
nb_train_samples =64
nb_validation_samples = 32
batch_size = 32
epochs = 200


model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

for layer in model.layers[:5]:
    layer.trainable = False

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(4096, activation="relu")(x)
x = Dense(2048, activation="relu")(x)
x = Dense(1024, activation="relu")(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(62, activation="softmax")(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)

# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

# Initiate the train and test generators with data Augumentation 
train_datagen = ImageDataGenerator(
rescale = 1./255)
# shear_range = 0.2,zoom_range = 0.2,
# horizontal_flip = True)
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
class_mode = None)

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width),
class_mode = None)

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#early = EarlyStopping(monitor='val_acc', min_delta=0, patience=100, verbose=1, mode='auto')


print(' Train the model' )
# model_final.fit_generator(
# train_generator,
# steps_per_epoch = 8000,
# #samples_per_epoch = nb_train_samples,
# epochs = 50,
# validation_data = validation_generator,
# #nb_val_samples = nb_validation_samples,
# validation_steps = 2000,
# callbacks = [checkpoint],
# class_weight=None, 
# #max_queue_size=50, 
# #workers=2, 
# use_multiprocessing=True, 
# shuffle=False,
# initial_epoch=0)
model_final.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),#variation of adam , momentum(optimizer)
              metrics=['accuracy'])
model_class = model_final.fit_generator(list(train_generator),
                         steps_per_epoch = 40,
                         epochs = 50,
                        validation_data = validation_generator,
                         validation_steps = 60,callbacks=[checkpoint])
# model_final.fit_generator(train_generator,
#  samples_per_epoch=8000,
#   epochs=50, verbose=1,
#    callbacks=[checkpoint],
#     validation_data=validation_generator,
#      validation_steps=2000,
#       class_weight=None, 
#       max_queue_size=50, 
#       workers=2, 
#       use_multiprocessing=True, 
#       shuffle=False,
#  initial_epoch=0)
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
# filenames = validation_generator.filenames
# nb_samples = len(filenames)
# print('total filesamples',nb_samples)
# predictions = model.predict_generator(validation_generator,nb_samples)
# print(predictions,"predictions first ")

# predictions = np.argmax(predictions, axis=-1) #multiple categories
# print(predictions,"predictions second ")




# label_map = (train_generator.class_indices)
# label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
# predictions = (label_map[k] for k in predictions)
# print(predictions,"predictions third")
