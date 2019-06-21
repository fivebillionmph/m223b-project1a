#!/usr/bin/env python

import os
import csv
#from skimage import io as skio
#import skimage
import matplotlib.pyplot as plt

"""
	images = loadImages(data_dir)
	analyzeImage(images[0])
"""

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
	return images

def plotImage(image_data):
	viewer = ImageViewer(image_data)
	viewer.show()

def plotImage(image_data, **kwargs):
	n_cols = 2
	n_rows = (len(image_data) // n_cols) + 1
	if len(image_data) % n_cols == 0:
		n_rows -= 1

	titles = kwargs["titles"] if "titles" in kwargs else None
	main_title = kwargs["main_title"] if "main_title" in kwargs else None

	i = 0
	done = False
	while not done:
		fig, axes = plt.subplots(1, n_cols, sharex=True, sharey=True, figsize=(13, 10))
		ax = axes.ravel()

		while True:
			ax[i % n_cols].imshow(image_data[i])
			if titles is not None:
				ax[i % n_cols].set_title(titles[i])
			elif main_title is not None:
				ax[i % n_cols].set_title(main_title)
			i += 1
			if i >= len(image_data):
				done = True
				break
			if i % n_cols == 0:
				break

		fig.show()

#skimage_filters = [
#	("gaussian", skimage.filters.gaussian),
#	("median", skimage.filters.median),
#	("sobel", skimage.filters.sobel),
#	("sobel_h", skimage.filters.sobel_h),
#	("sobel_v", skimage.filters.sobel_v),
#	("scharr", skimage.filters.scharr),
#	("scharr_h", skimage.filters.scharr_h),
#	("scharr_v", skimage.filters.scharr_v),
#	("prewitt", skimage.filters.prewitt),
#	("prewitt_h", skimage.filters.prewitt_h),
#	("prewitt_v", skimage.filters.prewitt_v),
#	("roberts", skimage.filters.roberts),
#	("roberts_pos_diag", skimage.filters.roberts_pos_diag),
#	("roberts_neg_diag", skimage.filters.roberts_neg_diag),
#	("laplace", skimage.filters.laplace),
#	("frangi", skimage.filters.frangi),
#	("hessian", skimage.filters.hessian),
#]

def cv2HistogramNorm(img, tile_size):
	img_new = img.copy()
	shape = img.shape
	width = shape[0]
	height = shape[1]
	depth = shape[2]
	expanse = tile_size - 1
	for i in range(width):
		for j in range(height):
			ix_left = i - expanse
			if ix_left < 0: ix_left = None

			ix_right = i + expanse
			if ix_right >= width: ix_right = None

			ix_top = j - expanse
			if ix_top < 0: ix_top = None

			ix_bottom = j + expanse
			if ix_bottom >= height: ix_bottom = None

			pixels = []
			if ix_top is not None and ix_left is not None and ix_right is not None:
				for k in range(ix_left, ix_right):
					pixels.append(img[k, ix_top])
			if ix_bottom is not None and ix_left is not None and ix_right is not None:
				for k in range(ix_left, ix_right):
					pixels.append(img[k, ix_bottom])
			if ix_left is not None and ix_top is not None and ix_bottom is not None:
				for k in range(ix_top, ix_bottom):
					pixels.append(img[ix_left, k])
			if ix_right is not None and ix_top is not None and ix_bottom is not None:
				for k in range(ix_top, ix_bottom):
					pixels.append(img[ix_right, k])

			if len(pixels) == 0:
				continue

			counter = 0
			sums = [0 for _ in range(depth)]
			for pixel in pixels:
				counter += 1
				for k in range(depth):
					sums[k] += pixel[k]
			for k in range(len(sums)):
				sums[k] /= counter
			new_val = img_new[i, j]
			s_new_val = sum(new_val)
			s_sums = sum(sums)
			if s_new_val / 2 > s_sums and s_new_val > len(new_val) * 100:
				for k in range(len(new_val)):
					new_val[k] = min(3 * new_val[k], 255)
			#for k in range(len(new_val)):
			#	new_val[k] -= sums[k]
			img_new[i, j] = new_val

	return img_new
