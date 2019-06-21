#!/usr/bin/env python

import sys
import os
import csv
from skimage import io as skio
from skimage.viewer import ViewImage
import matplotlib.pyplot as plt

def main():
	data_dir = sys.argv[1]
	images = loadImages(data_dir)
	analyzeImage(images[0])

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
			images.append(Image(line[1], skio.imread(os.path.join(data_dir, line[1])), True if line[2] == "yes" else False))
			break # DEBUG
	return images

def analyzeImage(image):
	fig = plt.figure()
	axis = plt.subplot(1, 1, 1)
	axis.imshow(image.data)
	axis.set_title(image.name)
	plt.show()

main()
