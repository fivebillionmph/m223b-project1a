import sqlite3
import argparse
from mod.helper import sqlite_dict_factory
from os.path import join as opj
from keras.models import load_model
from mod.anno import getImageData, getBounds
import cv2
import numpy as np

def main():
	cxn = sqlite3.connect("images.db")
	cxn.row_factory = sqlite_dict_factory
	cur = cxn.cursor()

	options = Options.getOptions()
	model, length, step_size = loadModel(options.results_dir)
	images = getImageData(cur)

	counter0 = 0
	counter1 = 0
	results = []
	counter = 0
	for image in images:
		training = False
		if image.trainable:
			training = True
		#elif image.has_feature:
		#	if counter1 >= 5:
		#		continue
		#	counter1 += 1
		#elif not image.has_feature:
		#	if counter0 >= 5:
		#		continue
		#	counter0 += 1
		counter += 1
		if counter % 10 == 0:
			print(counter)
		if not training:
			results.append(ImageResult(image, testImage(image, options, model, length, step_size)))

	#writeResultSquares(options, results)
	#writeResultImageScoreCutoff(options, results)
	CUTOFF = 0.9
	writeResultCount(options, results, cutoff = CUTOFF)
	##writeResultOverlappingBoxCutoff(options, results, cutoff = CUTOFF)
	##writeResultOverlappingExpBoxCutoff(options, results, cutoff = CUTOFF)
	##writeResultAverageDistanceCutoff(options, results, cutoff = CUTOFF)
	#writeDrawBoxesDestructive(options, results, cutoff = CUTOFF)

def testImage(image, options, model, length, step_size):
	img = image.getImg(options.images_dir)
	bounds = getBounds(img.shape[0], img.shape[1], length, step_size)
	bounds_img_list = []
	for b in bounds:
		sub_image = image.getSubImage(b, 1)
		sub_image.img = np.copy(sub_image.img)
		bounds_img_list.append((b, sub_image))
	counter = 0
	response_scores = []
	for bil in bounds_img_list:
		#img = np.array([bil[1].img.reshape((*bil[1].img.shape, 1))])
		img = np.array([bil[1].img])
		response = model.predict(img)
		response_scores.append(BoxResult(response[0][0], bil[0], image.masked(bil[0], 1)))
	return response_scores

def writeDrawBoxesDestructive(options, results, cutoff = 0.99):
	for res in results:
		for br in res.box_results:
			if br.pred_response > cutoff:
				res.image.destructiveDrawBounds(options.images_dir, br.box)
		cv2.imwrite(opj(options.results_dir, "test-annotated", res.image.name + "-" + str(cutoff) + ".png"), res.image.getImg(options.images_dir) * 255)

def writeResultSquares(options, results):
	with open(opj(options.results_dir, "test-annotated", "roc-squares.tsv"), "w") as f:
		f.write("response\tvalue\n")
		for res in results:
			for score in res.box_results:
				f.write("%d\t%f\n" % (1 if score.ground_response else 0, score.pred_response))

def writeResultImageScoreCutoff(options, results, cutoff = 0.0998850):
	with open(opj(options.results_dir, "test-annotated", "roc-" + str(cutoff) + ".tsv"), "w") as f:
		f.write("response\tvalue\n")
		for res in results:
			above_count = len([x for x in res.box_results if x.image > cutoff])
			f.write("%d\t%f\n" % (1 if res.image.has_feature else 0, above_count))

def writeResultCount(options, results, cutoff):
	cutoffs = [0.5, 0.9, 0.95, 0.98, 0.995]
	for cutoff in cutoffs:
		with open(opj(options.results_dir, "test-annotated", "scores-count-" + str(cutoff) + ".tsv"), "w") as f:
			f.write("response\tcount\n")
			for res in results:
				counts = len([x for x in res.box_results if x.pred_response > cutoff])
				f.write("%d\t%d\n" % (1 if res.image.has_feature else 0, counts))

def writeResultOverlappingBoxCutoff(options, results, cutoff = 0.99):
	with open(opj(options.results_dir, "test-annotated", "scores-overlapping-" + str(cutoff) + ".tsv"), "w") as f:
		f.write("response\tn_overlapping\n")
		for res in results:
			box_results = [x for x in res.box_results if x.pred_response > cutoff]
			counts = 0
			for i in range(len(box_results)):
				for j in range(len(box_results)):
					if i == j:
						continue
					if overlapping(box_results[i].box, box_results[j].box):
						counts += 1
			#norm = float(counts) / len(box_results)
			f.write("%d\t%d\n" % (1 if res.image.has_feature else 0, counts))

def writeResultOverlappingExpBoxCutoff(options, results, cutoff = 0.99):
	with open(opj(options.results_dir, "test-annotated", "scores-overlapping-exp-" + str(cutoff) + ".tsv"), "w") as f:
		f.write("response\tn_overlapping\n")
		for res in results:
			box_results = [x for x in res.box_results if x.pred_response > cutoff]
			counts = 0
			for i in range(len(box_results)):
				this_counts = 0
				for j in range(len(box_results)):
					if i == j:
						continue
					if overlapping(box_results[i].box, box_results[j].box):
						this_counts += 1
				counts += this_counts ** 2
			#norm = float(counts) / len(box_results)
			f.write("%d\t%d\n" % (1 if res.image.has_feature else 0, counts))

def writeResultAverageDistanceCutoff(options, results, cutoff = 0.99, min_count = 5):
	with open(opj(options.results_dir, "test-annotated", "scores-distances-" + str(cutoff) + ".tsv"), "w") as f:
		f.write("response\taverage_distance\n")
		for res in results:
			response_int = 1 if res.image.has_feature else 0
			box_results = [x for x in res.box_results if x.pred_response > cutoff]
			if len(box_results) < min_count:
				f.write("%d\t%f\n" % (response_int, 0.0))
				continue
			distances = []
			for i in range(len(box_results) - 1):
				for j in range(i + 1, len(box_results)):
					distances.append(boxDistance(box_results[i].box, box_results[j].box))
			average_distance = sum(distances) / len(distances)
			f.write("%d\t%f\n" % (response_int, 1 / average_distance))

class ImageResult:
	def __init__(self, image, box_results):
		self.image = image
		self.box_results = box_results

class BoxResult:
	def __init__(self, pred_response, box, ground_response):
		self.pred_response = pred_response
		self.box = box
		self.ground_response = ground_response

class Options:
	def __init__(self, args):
		self.images_dir = args.images_dir
		self.results_dir = args.results_dir

	@classmethod
	def getOptions(cls):
		parser = argparse.ArgumentParser()
		parser.add_argument("images_dir")
		parser.add_argument("results_dir")

		args = parser.parse_args()
		return cls(args)

def loadModel(results_dir):
	model = load_model(opj(results_dir, "model"))

	with open(opj(results_dir, "options.txt")) as f:
		for line in f:
			if "bounding size width:" in line:
				length = int(line.split(": ")[1])
			elif "step size: " in line:
				step_size = int(line.split(": ")[1])

	return model, length, step_size

def overlapping(box1, box2):
	# https://www.geeksforgeeks.org/find-two-rectangles-overlap/
	l1 = (box1[0], box1[2])
	r1 = (box1[1], box1[3])
	l2 = (box2[0], box2[2])
	r2 = (box2[1], box2[3])
	if l1[0] > r2[0] or l2[0] > r1[0]:
		return False
	if l1[1] > r2[1] or l2[1] > r1[1]:
		return False
	return True

def boxDistance(box1, box2):
	l1 = (float(box1[0]), float(box1[2]))
	l2 = (float(box2[0]), float(box2[2]))

	return (((l1[0] - l2[0])**2) + ((l1[1] - l2[1])**2))**0.5

main()
