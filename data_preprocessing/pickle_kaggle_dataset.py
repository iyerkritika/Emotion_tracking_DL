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
immatrix = []
label_final = []
print "Start"
for i in range(6):
	print "Processing ", i
	# Obtain image directory information
	images_path = 'data/' + str(i) + '/'
	images = glob.glob(images_path + '*.png')

	# Get lavels
	label = np.empty(len(images))
	label.fill(i)

	# Add images to final large image
	for image in images:
		img = cv2.imread(image)
		immatrix.append(img)

	print len(label)

	# Append results to final area
	label_final = np.append(label_final, label)
	print len(label_final)
	print len(immatrix)

print "Pickling"	
file = open("database.pickle", "wb")
pickle.dump((immatrix, label_final), file)
print "Done!"








# file = open('database.pickle', 'wb')










