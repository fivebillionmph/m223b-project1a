import sys
import os
import random
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers.core import Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt

TRAINING_P = 0.7
EPOCHS = 15

class Image:
	def __init__(self, name, data, response):
		self.name = name
		self.data = data
		self.response = response

def loadImages(data_dir):
	labels_file = os.path.join(data_dir, "Labels.csv")
	images = []
	with open(labels_file) as f:
		reader = csv.reader(f)
		next(reader) # skip header row
		for line in reader:
			image_data = cv2.imread(os.path.join(data_dir, line[1]))
			image_data = cv2.resize(image_data, (512, 512))
			image_data = filterImage(image_data)
			image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
			#image_data = cheat(image_data, True if line[2] == "yes" else False)
			image_data = img_to_array(image_data)
			image_data = image_data / 255
			images.append(Image(line[1], image_data, True if line[2] == "yes" else False))
	return images

def filterImage(img, lower_bound=240):
	blur = cv2.GaussianBlur(img, (15, 15), 2)
	lower = np.array([lower_bound, lower_bound, lower_bound])
	upper = np.array([255, 255, 255])
	mask = cv2.inRange(blur, lower, upper)
	masked_img = cv2.bitwise_and(img, img, mask=mask)
	return masked_img

def model1():
	model = Sequential()

	# first set
	model.add(Conv2D(20, (5, 5), padding="same", input_shape=(512, 512, 1)))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	# second set of CONV => RELU => POOL layers
	model.add(Conv2D(50, (5, 5), padding="same"))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	# first (and only) set of FC => RELU layers
	model.add(Flatten())
	model.add(Dense(500))
	model.add(Activation("relu"))
	# softmax classifier
	model.add(Dense(1))
	model.add(Activation("softmax"))
	model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")

	return model

def model2():
	model = Sequential()

	# input layer
	model.add(Dense(50, activation="relu", input_shape=(512,512,1)))

	# hidden layers
	model.add(Dropout(0.3, noise_shape=None, seed=None))
	model.add(Dense(30, activation="relu"))
	model.add(Dropout(0.2, noise_shape=None, seed=None))
	model.add(Flatten())
	model.add(Dense(20, activation="relu"))

	# output layer
	model.add(Dense(1, activation="sigmoid"))

	model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

	return model

def model3():
	model = Sequential()

	# input layer
	model.add(Dense(50, activation="relu", input_shape=(512,512,1)))

	# hidden layers
	model.add(Dropout(0.3, noise_shape=None, seed=None))
	model.add(Dense(30, activation="relu"))
	model.add(Flatten())

	# output layer
	model.add(Dense(1, activation="sigmoid"))

	model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

	return model

images = loadImages(sys.argv[1])
training_size = int(len(images) * TRAINING_P)

random.shuffle(images)
training_images = images[0:training_size]
test_images = images[training_size:len(images)]
X = [x.data for x in training_images]
y = [1 if y.response else 0 for y in training_images]
X = np.array(X)
y = np.array(y)

model = model2()
print(model.summary())
history = model.fit(X, y, validation_split=0.3, epochs=EPOCHS)
model.save("../data/model-test")

try:
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, EPOCHS), history.history["loss"], label="train_loss")
	plt.plot(np.arange(0, EPOCHS), history.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, EPOCHS), history.history["acc"], label="train_acc")
	plt.plot(np.arange(0, EPOCHS), history.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy on Santa/Not Santa")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig("../data/model-test-plot.png")
except:
	print("could not make plot")

#model = load_model("../data/model-test")

sys.stdout.write("name\tcorrect\tpredition\ttruth\tnegative\tpositive\n")
correct_count = 0
for ti in test_images:
	test_image = ti.data
	res = model.predict(np.array([test_image]))
	pos = res[0][0]
	neg = 1 - pos
	if neg > pos:
		has_feature = False
	else:
		has_feature = True
	if has_feature == ti.response:
		correct = True
		correct_count += 1
	else:
		correct = False
	sys.stdout.write("%s\t%s\t%s\t%s\t%f\t%f\n" % (ti.name, str(correct), str(has_feature), str(ti.response), neg, pos))
sys.stdout.write("-" * 100)
sys.stdout.write("\n")
sys.stdout.write("%d correct out of %d\n" % (correct_count, len(test_images)))
