# https://pypi.org/project/labelImg/
#   <object-class> <x> <y> <width> <height>
# 
# kaggle.com | efficientdet | Model Variations: 
#   TensorFlow Lite, 
#   variation: lite2-detection-metadata
#  
# !pip install tflite-model-maker
# !pip install tflite-support
# pip install pyyaml h5py  # Required to save models in HDF5 format

#https://medium.com/analytics-vidhya/image-dataset-labeling-annotation-bec3390eda2d
#https://github.com/Qengineering/TensorFlow_Lite_SSD_RPi_64-bits
#https://www.kaggle.com/models/google/mobilenet-v2

import numpy as np 
import tensorFlow as tf
import keras
import os

shape=(150,150,3)
classNames=[]
directory='''PATH TO DATASET'''

with open('classlist.txt') as file:
    classNames=[line.rstrip() for line in file]

#https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
new_dataset=tf.keras.utils.image_dataset_from_directory(
    directory,
    labels='inferred',
    label_mode='int',
    class_names=classNames,
    color_mode='rgb',
    batch_size=32,
    image_size=(320,640),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
    #**kwargs
)
size=(320,320)
new_dataset=new_dataset.map(lambda x,y:(tf.image.resize(x,size),y))

# m = tf.keras.Sequential([
#     hub.KerasLayer("https://www.kaggle.com/models/google/mobilenet-v2/frameworks/TensorFlow2/variations/035-128-classification/versions/2")
# ])
# base_model=keras.applications.Xception(
#     weights=m,
#     input_shape=shape,
#     include_top=False
# )

base_model=keras.applications.Xception(
    weights='AlexNet',
    input_shape=shape,
    include_top=False
)
base_model.trainable=False

inputs=keras.Input(shape=(shape))
x=base_model(inputs,training=False)
x=keras.layers.GlobalAveragePooling2D()(x)
outputs=keras.layers.Dense(1)(x)
model=keras.Model(inputs,outputs)
# model.summary()

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])
model.fit(new_dataset,epochs=20,validation_data=.2)

model.save('my_model.keras')
