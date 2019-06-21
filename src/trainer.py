import sqlite3
import argparse
from mod.helper import sqlite_dict_factory
import json
import cv2
import os
from keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import random
from mod.anno import Image, TrainableImage, getBounds, getImageData
from mod.models import models

BATCH_SIZE = 32

def main():
	cxn = sqlite3.connect("images.db")
	cxn.row_factory = sqlite_dict_factory

	c = cxn.cursor()
	options = parseArgs()
	images = getImageData(c)
	training_set_images = createTrainingSet(images, options)
	training_set_images = evenResponse(training_set_images)
	model = models["binary2"](training_set_images[0].img.shape)
	print(model.summary())
	writeModel(model, os.path.join(options.results_dir, "model-summary.txt"), os.path.join(options.results_dir, "model.png"))
	print("training on %d images (%d images with response)" % (len(training_set_images), len([tsi for tsi in training_set_images if tsi.response])))
	print(training_set_images[0].img.shape)
	nn_model, history = trainImages(model, training_set_images, options.validation_size, options.epochs)
	nn_model.save(os.path.join(options.results_dir, "model"))
	graphModel(history, os.path.join(options.results_dir, "model-plot.png"), options.epochs)
	options.write("options.txt")

	#cxn.commit()
	cxn.close()

class Options:
	def __init__(self, images_dir, results_dir, width, step_size, min_pixels, validation_size, epochs):
		self.images_dir = images_dir
		self.results_dir = results_dir
		self.width = width
		self.step_size = step_size
		self.min_pixels = min_pixels
		self.validation_size = validation_size
		self.epochs = epochs

	def write(self, name):
		filename = os.path.join(self.results_dir, name)
		with open(filename, "w") as f:
			f.write("bounding size width: %d\n" % (self.width,))
			f.write("step size: %d\n" % (self.step_size,))
			f.write("minimun pixels: %d\n" % (self.min_pixels,))
			f.write("validation size: %f\n" % (self.validation_size,))
			f.write("number of epochs: %d\n" % (self.epochs,))

def parseArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument("images_dir", help="directory where images are stored")
	parser.add_argument("results_dir", help="directory to save results")
	parser.add_argument("-w", "--width", help="box width (and length)", type=int, default=50)
	parser.add_argument("-s", "--stepsize", help="step size", type=int, default=20)
	parser.add_argument("-p", "--pixels", help="minimum number of pixels in box to count as true", type=int, default=1)
	parser.add_argument("-v", "--validation", help="validation set size (set between 0 and 1)", type=float, default=0.3)
	parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=15)

	args = parser.parse_args()
	return Options(args.images_dir, args.results_dir, args.width, args.stepsize, args.pixels, args.validation, args.epochs)

def createTrainingSet(images, options):
	tr_images = [img for img in images if img.trainable]
	augments = []
	for image in tr_images:
		img = image.getImg(options.images_dir)
		bounds = getBounds(img.shape[0], img.shape[1], options.width, options.step_size)
		for b in bounds:
			sub_image = image.getSubImage(b, options.min_pixels, continuous = False)
			augments.append(sub_image)
			if sub_image.response:
				augments.append(sub_image.rotate(1))
				augments.append(sub_image.rotate(2))
				augments.append(sub_image.rotate(3))
	#for tsi in augments:
	#	tsi.img = cv2.resize(tsi.img, (100, 100))
	#for tsi in augments:
	#	tsi.img = tsi.img.reshape((*tsi.img.shape, 1))
	return augments

def trainImages(model, training_set_images, validation_size, epochs):
	X = np.array([tsi.img for tsi in training_set_images])
	y = np.array([tsi.response for tsi in training_set_images])
	history = model.fit(X, y, validation_split=validation_size, epochs=epochs, batch_size=BATCH_SIZE)
	return model, history

def graphModel(history, filename, epochs):
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
	plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, epochs), history.history["acc"], label="train_acc")
	plt.plot(np.arange(0, epochs), history.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy")                                                                                                                                          
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(filename)

def evenResponse(training_set_images):
	pos = [tsi for tsi in training_set_images if tsi.response]
	neg = [tsi for tsi in training_set_images if not tsi.response]
	pos.extend(random.sample(neg, len(pos)))
	random.shuffle(pos)
	return pos

def evenResponse1(training_set_images):
	pos = [tsi for tsi in training_set_images if tsi.response]
	neg = [tsi for tsi in training_set_images if not tsi.response]
	while len(pos) < len(neg):
		pos.extend(random.sample(pos, 100))
	neg.extend(pos)
	random.shuffle(neg)
	return neg

def writeModel(model, summary_file, model_file):
	with open(summary_file, "w") as f:
		model.summary(print_fn=lambda x: f.write(x))

	plot_model(model, to_file=model_file)

main()
