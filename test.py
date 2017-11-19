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

data=[]
print "Start"
for i in range(6):
	print "Processing ", i
	# Obtain image directory information
	images_path = 'data/' + str(i) + '/'
	images = glob.glob(images_path + '*.png')

	# Get lavels
	# label = np.empty(len(images))
	# label.fill(i)

	# Add images to final large image
	for image in images:
		img = cv2.imread(image)
		data_both=img,i
		data.append(data_both)

print "Pickling"
file = open("database.pickle", "wb")
pickle.dump(data, file)
print "Done!"
