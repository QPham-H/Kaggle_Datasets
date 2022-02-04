# Code by Quoc Pham for Kaggle's Pneumonia Detection Dataset
import tensorflow as tf
import os
from keraslayers import Dense, Dropout, Input, Flatten
from keras.models import Sequential, Model

# We are using transfer learning aka a pre-trained cnn model with frozen weights, so we will resize our images to fit the input size
epoch_num = 25
batch_s = 32 # Batch sizes should be of a power of 2 that fits the memory requirement of the device
img_ht =  224 # Height and width to match inputs of pre-trained model
img_wd = 224

# Loading dataset and pre-processing
# Note that folder structure has images already separated into two subfolders in the above directories
# Utilize the directory structure to our advantage and create a tf Dataset object for ease of use
train_data = tf.preprocessing.image_dataset_from_directory(
	'/input/chest-xray-pneumonia/chest_xray/train',
	image_size = (img_ht, img_wd),
	batch_size = batch_s,
)
	
val_data = tf.preprocessing.image_dataset_from_directory(
	'/input/chest-xray-pneumonia/chest_xray/val',
	image_size = (img_ht, img_wd),
	batch_size = batch_s,
)

test_data = tf.preprocessing.image_dataset_from_directory(
	'/input/chest-xray-pneumonia/chest_xray/test',
	image_size = (img_ht, img_wd),
	batch_size = batch_s,
)

# Checking the size of the training data for the binary classification 
print("Number of NORMAL training data: ")
print(len(os.listdir('/input/chest-xray-pneumonia/chest_xray/train/NORMAL')))
print("Number of PNEUMONIA training data: ")
print(len(os.listdir('/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA')))

# Using data augmentation (by rotations) to oversample the minority set at the risk of overfitting

# Building the model by transfer learning, i.e. pre-trained weights for the CNN but with additional custom FFNN
base_model = tf.keras.applications.ResNet50(
	include_top = False,
	weights = 'imagenet',
	pooling = 'max' # Global Max Pooling to be applied
)

print(base_model.summary())

# Freeze the pre-trained weights in the CNN layers
for layers in base_model.layers:
	layer.trainable = False
	
# Adding Input layer to be pre-processed by ResNet50, i.e. normalize by subtracting the mean of the imagenet data set
input = Input([img_ht, img_wd, 3], dtype = tf.uint8)
x = tf.keras.applications.resnet50.preprocess_input(input)
x = base_model(x)

# Adding the FFNN layers
FFNN_model = Sequential()
FFNN_model.add(Dense(1024, activation='relu'))
FFNN_model.add(Dropout(0.2))
FFNN_model.add(Dense(512, activation='relu'))
FFNN_model.add(Dropout(0.2))
FFNN_model.add(Dense(1, activation='sigmoid')) # sigmoid for binary and softmax for more than 2 classes

# Combining the FFNN model together with the base model
x = FFNN_model(x)
model = Model(inputs = [input], outputs = [x])
		
# Training the model (future edits to include Kfold CV)
metrics = [
	tf.keras.metrics.BinaryAccuracy(name='accuracy'),
	tf.keras.metrics.Precision(name='precision'),
	tf.keras.metrics.Recall(name='recall'),
	tf.keras.metrics.AUC(name='AUC')
]

# This stops training the model if the monitored metric is no longer change in some number of epochs (patience)
# For this medical classification, we value recall over precision. That is, we prefer to minimize false negatives
early_stopping = tf.keras.callbacks.EarlyStopping(
	monitor = 'val_recall',
	mode = 'max',
	patience = 5
)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics) # Adaptive Momentum Estimation aka Adam is the latest best performing gradient descent algorithm

history = model.fit(
	train_data,
	validation_data = val_data,
	epochs = epoch_num,
	callbacks = [early_stopping]
)

# Evaluating the model
print(model.evaluate(
	test_data,
	batch_size = batch_size
))
