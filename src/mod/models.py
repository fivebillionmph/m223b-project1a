from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout
from keras.layers.core import Flatten, Activation
from keras.optimizers import SGD
import numpy as np

def _model1(shape):
	model = Sequential()

	model.add(Dense(50, activation="relu", input_shape=shape))
	model.add(Dropout(0.3, noise_shape=None, seed=None))
	model.add(Dense(30, activation="relu"))
	model.add(Flatten())
	model.add(Dense(1, activation="sigmoid"))

	model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
	return model

def _model2(shape):
	model = Sequential()

	model.add(Dense(50, activation="relu", input_shape=shape))
	model.add(Dropout(0.3, noise_shape=None, seed=None))
	model.add(Dense(30, activation="relu"))
	model.add(Dropout(0.2, noise_shape=None, seed=None))
	model.add(Flatten())
	model.add(Dense(20, activation="relu"))

	# output layer
	model.add(Dense(1, activation="sigmoid"))
	
	model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

	return model

def _model3(shape):
	# from here:
	# https://keras.io/getting-started/sequential-model-guide/
	model = Sequential()

	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=shape))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["accuracy"])

	return model

def _model4(shape):
	import keras
	from keras.layers import Input
	input_tensor = Input(shape=shape)  # this assumes K.image_data_format() == 'channels_last'
	base = keras.applications.vgg16.VGG16(include_top=False, input_tensor=input_tensor, input_shape=shape, pooling=None, classes=2)
	#base = keras.applications.densenet.DenseNet121(include_top=False, input_tensor=input_tensor, input_shape=shape, pooling=None, classes=2)
	model = Sequential()
	model.add(base)
	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
	return model

def _model5(shape):
	# from here:
	# https://keras.io/getting-started/sequential-model-guide/
	model = Sequential()

	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=shape))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	# extra
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	# /extra

	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["accuracy"])

	return model

def _model6(shape):
	model = Sequential()

	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=shape))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation="tanh"))

	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mean_squared_error', optimizer="adam", metrics=["accuracy"])

	return model

def _model7(shape):
	# from: https://medium.com/@kylepob61392/airplane-image-classification-using-a-keras-cnn-22be506fdb53
	
	# Define hyperparamters
	n_layers = 5
	MIN_NEURONS = 20
	MAX_NEURONS = 120
	KERNEL = (3, 3)
	
	# Determine the # of neurons in each convolutional layer
	steps = np.floor(MAX_NEURONS / (n_layers + 1))
	nuerons = np.arange(MIN_NEURONS, MAX_NEURONS, steps)
	nuerons = nuerons.astype(np.int32)
	
	# Define a model
	model = Sequential()
	
	# Add convolutional layers
	for i in range(0, n_layers):
		if i == 0:
			model.add(Conv2D(nuerons[i], KERNEL, input_shape=shape))
		else:
			model.add(Conv2D(nuerons[i], KERNEL))
		model.add(Activation('relu'))
		model.add(Dropout(0.50))
	
	# Add max pooling layer
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(MAX_NEURONS))
	model.add(Activation('relu'))
	
	# Add output layer
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	
	# Compile the model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	# Print a summary of the model
	model.summary()

	return model

models = {
	"1": _model1,
	"2": _model2,
	"binary": _model3,
	"keras": _model4,
	"binary-extra-layer": _model5,
	"continuous": _model6,
	"binary2": _model7,
}
