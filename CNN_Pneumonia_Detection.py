# Code by Quoc Pham for Kaggle's Pneumonia Detection Dataset
import tensorflow as tf
import os


# We are using transfer learning aka a pre-trained cnn model with frozen weights, so we will resize our images to fit the input size
epoch_num = 25
batch_s = 32 # Batch sizes should be of a power of 2 that fits the memory requirement of the device
img_ht =  # Height and width to match inputs of pre-trained model
img_wd = 

# Loading dataset and pre-processing
# Note that folder structure has images already separated into two subfolders in the above directories
# Utilize the directory structure to our advantage and create a tf Dataset object for ease of use
train_data = tf.preprocessing.image_dataset_from_directory(
	'/input/chest-xray-pneumonia/chest_xray/train',
	
)
	
val_data = tf.preprocessing.image_dataset_from_directory(
	'/input/chest-xray-pneumonia/chest_xray/val',
	
)

test_data = tf.preprocessing.image_dataset_from_directory(
	'/input/chest-xray-pneumonia/chest_xray/test',
	
)

# Checking the size of the training data for the binary classification 
print("Number of NORMAL training data: ")
print(len(os.listdir('/input/chest-xray-pneumonia/chest_xray/train/NORMAL')))
print("Number of PNEUMONIA training data: ")
print(len(os.listdir('/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA')))

# Using data augmentation to oversample the minority set at the risk of overfitting

# Building the model by transfer learning, i.e. pre-trained weights for the CNN but with additional custom FFNN
model = tf.keras.applications.ResNet50(
	include_top = False,
	weights = 'imagenet',
	input_shape = [],
	pooling = 'max'
)

# Adding the FFNN layers
model.add(tf.keras.layers.Flatten)
		
# Training the model with Kfold CV
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
	batch_size = batch_s,
	callbacks = [early_stopping],

)

# Evaluating the model
print(model.evaluate(
	test_data,
	batch_size = batch_size
))
