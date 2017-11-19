###########################################################
# Deep Learning for Advanced Robot Perception
# Final Project: Emotion Recognition
# Augment Data
# prepared by: Katie Gandomi and Kritika Iyer
###########################################################
import os
import cv2
import glob
import numpy as np

maximages = 6000

tempNum = 1117000 # Use a really large number so we dont have conflicting file names

for i in range(6):
	images_path = 'data/' + str(i) + '/'
	images = glob.glob(images_path + '*.png')

	newnum = len(images)
	while newnum < maximages :
		r = np.random.randint(newnum)
		path = images_path+'image'+str(r)+'.png'

		if(os.path.isfile(path)):
			img=cv2.imread(path)
			img = cv2.flip(img, 1)
			cv2.imwrite(images_path+'image'+str(tempNum) + '.png', img)
			tempNum = tempNum + 1
			newnum = newnum + 1

