###########################################################
# Deep Learning for Advanced Robot Perception
# Final Project: Emotion Recognition
# Script for preparing Kaggle Dataset
# prepared by: Katie Gandomi and Kritika Iyer
###########################################################

import os
import csv
import numpy as np
from PIL import Image

csv_file = 'fer2013/fer2013.csv'
data_directory = 'data/'
img_width = 48
img_height = 48
img_counter = 0

img_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
with open(csv_file, 'rb') as csvfile:
	print "Processing Kaggle Data..."
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader :
		if img_counter != 0:
			sub_directory = data_directory + str(row[0])

			if not os.path.exists(sub_directory):
				os.makedirs(sub_directory)

			data = np.zeros((48, 48, 3), dtype=np.uint8)
			k_data = [int(i) for i in row[1].split()]

			r = 0;
			for i in range(len(k_data)):
				if i%48 == 0 and i != 0:
					r += 1

				c = i % 48
				data[r][c] = k_data[i]

			img = Image.fromarray(data)
			img.save(sub_directory + '/image' + str(img_dict[int(row[0])]) + '.png')

			img_dict[int(row[0])] = img_dict[int(row[0])] + 1

		img_counter +=1

print "Done !"






