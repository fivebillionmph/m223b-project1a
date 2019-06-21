import json
import os
import cv2
import numpy as np

class Image:
	def __init__(self, name, has_feature, trainable, mask):
		self.name = name
		self.has_feature = True if has_feature == 1 else False
		self.trainable = True if trainable == 1 else False
		if(mask is None):
			self.mask = None
		else:
			self.mask = set()
			raw_mask = json.loads(mask)
			for point_name in raw_mask["points"]:
				point = raw_mask["points"][point_name]
				self.mask.add((int(point[0]), int(point[1])))
		self.img = None

	def getImg(self, images_dir):
		if self.img is None:
			file_name = os.path.join(images_dir, self.name)
			img = cv2.imread(file_name)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = img.reshape((*img.shape, 1))
			### filters
			layer2 = lowIntensityFilter(img)
			layer3 = lowIntensityFilter(alphaBetaFilter(img, 1, 90))
			img = np.dstack((img, layer2))
			img = np.dstack((img, layer3))
			###
			img = img / 255
			self.img = img
		return self.img

	def getSubImage(self, bounds, min_pixels, continuous = False):
		sub_img = self.img[bounds[0]:bounds[1], bounds[2]:bounds[3], :]

		if continuous:
			response = self.maskedPercent(bounds)
		else:
			response = self.masked(bounds, min_pixels)

		return TrainableImage(sub_img, response)

	def masked(self, bounds, min_pixels):
		if self.mask is None:
			return False
		counter = 0
		for m in self.mask:
			if m[1] >= bounds[0] and m[1] < bounds[1] and m[0] >= bounds[2] and m[0] < bounds[3]:
				counter += 1
				if counter >= min_pixels:
					return True
		return False

	def maskedPercent(self, bounds):
		if self.mask is None:
			return 0.0
		counter = 0
		for m in self.mask:
			if m[1] >= bounds[0] and m[1] < bounds[1] and m[0] >= bounds[2] and m[0] < bounds[3]:
				counter += 1
		total = (bounds[3] - bounds[2]) * (bounds[1] - bounds[0])
		return counter / total

	def writeWithMask(self, image_dir, filename):
		img = self.getImg(image_dir)
		if self.mask is not None:
			for m in self.mask:
				img[m[1], m[0]] = 0
		cv2.imwrite(filename, img)

	def writeVerticalLine(self, image_dir, filename):
		img = self.getImg(image_dir)
		for i in range(len(img[:,100])):
			for j in range(95,105):
				img[i,j] = 0
		cv2.imwrite(filename, img)

	def writeBoundsWithResponse(self, image_dir, filename, size, min_pixels):
		img = self.getImg(image_dir)
		bounds = getBounds(img.shape[0], img.shape[1], size, size)
		for b in bounds:
			if self.masked(b, min_pixels):
				self.destructiveDrawBounds(image_dir, b)
		cv2.imwrite(filename, img)

	def destructiveDrawBounds(self, image_dir, bound):
		img = self.getImg(image_dir)
		cv2.rectangle(img, (bound[2], bound[0]), (bound[3], bound[1]), (1, ), 1)
		return
		for i in range(bound[0], bound[1]):
			for j in range(bound[2], bound[3]):
				img[i,j] = 1

class TrainableImage:
	def __init__(self, img, response):
		self.img = img
		self.response = response

	def rotate(self, e):
		if e == 1:
			return TrainableImage(cv2.rotate(self.img, cv2.ROTATE_90_CLOCKWISE), self.response)
		elif e == 2:
			return TrainableImage(cv2.rotate(self.img, cv2.ROTATE_180), self.response)
		elif e == 3:
			return TrainableImage(cv2.rotate(self.img, cv2.ROTATE_90_COUNTERCLOCKWISE), self.response)
		else:
			raise Exception("rotation e must be in [1,3]")

def getBounds(width, height, size, step_size):
	bounds = []

	# forward width
	# forward height
	start_width = 0
	start_height = 0
	while True:
		end_width = start_width + size
		end_height = start_height + size
		if end_width > width:
			start_width = 0
			end_width = start_width + size
			start_height += step_size
			end_height = start_height + size
		if end_height > height:
			break
		bounds.append((start_width, end_width, start_height, end_height))
		start_width += step_size

	# this will miss the right and bottom edges, but that's okay for now
	return bounds

def getImageData(cursor):
	cursor.execute("select * from images")
	res = cursor.fetchall()
	images = []
	for row in res:
		images.append(Image(row["name"], row["has_feature"], row["trainable"], row["feature_mask"]))
	return images

def lowIntensityFilter(img, lower_bound=240):
	blur = cv2.GaussianBlur(img, (15, 15), 2)
	lower = np.array([lower_bound])
	upper = np.array([255])
	mask = cv2.inRange(blur, lower, upper)
	masked_img = cv2.bitwise_and(img, img, mask=mask)
	return masked_img

def alphaBetaFilter(img, alpha, beta):
	new_img = np.zeros(img.shape, img.dtype)
	for y in range(img.shape[0]):
		for x in range(img.shape[1]):
			for c in range(img.shape[2]):
				new_img[y,x,c] = np.clip(alpha * img[y,x,c] + beta, 0, 255)
	return new_img
