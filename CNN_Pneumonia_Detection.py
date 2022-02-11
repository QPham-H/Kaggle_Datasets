"""
Using a pre-trained model for Pneumonia Detection
Coded by Quoc Pham
02.2022
"""

import tensorflow as tf
import os
import cv2

from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input, Flatten, MaxPooling2D, Conv2D
from tensorflow.keras.models import Sequential, Model

# Batch sizes should be of a power of 2 that fits the memory requirement of the device
BATCH = 32
EPOCHS = 1

##img = cv2.imread('chest_xray/train/NORMAL/IM-0115-0001.jpeg')
##print('This is the image shape: ')
##print(img.shape)

# Height and width to match inputs of pre-trained model
img_ht = 128 # Original ht is 1858
img_wd = 128 # Original wd is 2090

# Loading dataset and pre-processing
# Note that folder structure has images already separated into two subfolders in the directories
# Use the directory structure to our advantage and create a tf Dataset object for ease of use
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    'chest_xray/train',
    image_size = (img_ht, img_wd),
    color_mode = 'grayscale', # Reduce from 3 channels to 1 since xrays are already black and white and for speed and better generalizing
    batch_size = BATCH,
    #crop_to_aspect_ratio = True # Seem to get better results when this is False
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    'chest_xray/val',
    image_size = (img_ht, img_wd),
    color_mode = 'grayscale',
    batch_size = BATCH,
    #crop_to_aspect_ratio = True
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    'chest_xray/test',
    image_size = (img_ht, img_wd),
    color_mode = 'grayscale',
    batch_size = BATCH,
    #crop_to_aspect_ratio = True
)

# Checking the size of the training data for the binary classification 
Normal_ct = len(os.listdir('chest_xray/train/NORMAL'))
print("Number of NORMAL training data: ")
print(Normal_ct)

Pneumonia_ct = len(os.listdir('chest_xray/train/PNEUMONIA'))
print("Number of PNEUMONIA training data: ")
print(Pneumonia_ct)

# We will use class weights in the fit function since the dataset is particularly imbalanced
# A better iteration would be to use data augmentation on the minority class
Normal_wt = (Normal_ct + Pneumonia_ct)/(2 * Normal_ct)
Pneumonia_wt = (Normal_ct + Pneumonia_ct)/(2 * Pneumonia_ct)

class_weights = {
    0: Normal_wt,
    1: Pneumonia_wt
    }

# Build the model
def build_model():
    input_layer = Input(shape=(img_ht, img_wd,1))
    x = BatchNormalization()(input_layer)
    x = Conv2D(8,3,activation='relu',padding='same')(x)
    x = MaxPooling2D(2)(x)
    x = Conv2D(16,3,activation='relu',padding='same')(x)
    x = MaxPooling2D(2)(x)
    x = Conv2D(32,3,activation='relu',padding='same')(x)
    x = MaxPooling2D(2)(x)
    x = Flatten()(x)
    
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(512,activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(1,activation='sigmoid')(x) 
    model = Model(input_layer, out)
    print(model.summary())

    return model

# Training the model (next version to include Kfold CV)
METRICS = [
	tf.keras.metrics.BinaryAccuracy(name='accuracy'),
	tf.keras.metrics.Precision(name='precision'),
	tf.keras.metrics.Recall(name='recall'),
	tf.keras.metrics.AUC(name='AUC')
]

### This stops training the model if the monitored metric is no longer change in some number of epochs (patience)
### For this medical classification, we value recall over precision. That is, we prefer to minimize false negatives
##early_stopping = tf.keras.callbacks.EarlyStopping(
##	monitor = 'val_recall',
##	mode = 'max',
##	patience = 5
##)


model = build_model()

print(model.summary())

# Compile the model
model.compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = METRICS
    )

# Train model on split
history = model.fit(
    train_data,
    validation_data = val_data,
    epochs = EPOCHS,
    class_weight = class_weights
)


# Final evaluation on held out set
loss, accuracy, precision, recall, AUC = model.evaluate(
    test_data,
    batch_size = BATCH
    )

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'AUC: {AUC}')

