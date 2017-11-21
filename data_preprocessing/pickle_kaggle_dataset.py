###########################################################
# Deep Learning for Advanced Robot Perception
# Final Project: Emotion Recognition
# Script for preparing pickled kaggle data
# prepared by: Katie Gandomi and Kritika Iyer
###########################################################

import glob
import pickle
import cv2
from PIL import Image 
import numpy as np

# immatrix = Image.new((48,48*35887))
# immatrix = []
immatrix = np.empty((36000, 48,48), int)
label_final = []
print "Start"
count = 0
for i in range(6):
	print "Processing ", i
	# Obtain image directory information
	images_path = '../../data/' + str(i) + '/'
	images = glob.glob(images_path + '*.png')

	# Get lavels
	label = np.empty(len(images))
	label.fill(i)

	# Add images to final large image
	for image in images:
		img = cv2.imread(image, 0)
		# print img.shape
		immatrix[count] = img
		count = count + 1
		# immatrix = np.append(immatrix, np.asarray(img), axis=0)
		# print immatrix.shape

		# immatrix.append(label)
		# np.concatenate((immatrix, img))

	# Append results to final area
	label_final = np.append(label_final, label)
	print label_final.shape
	print immatrix.shape

print "Pickling"	
file1 = open("database.pickle", "wb")
pickle.dump((immatrix, label_final), file1)
print "Done!"

# print len(immatrix[0])
# print len(immatrix[0][0])







# file = open('database.pickle', 'wb')










