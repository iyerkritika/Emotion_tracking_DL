###########################################################
# Deep Learning for Advanced Robot Perception
# Project - Emotion detector
# Convolutional Neural Network for Emotion Recognition
# prepared by: Katie Gandomi, Kritika Iyer
###########################################################

# Import dependencies
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.models import model_from_json
import cv2
import numpy as np

# load json and create model
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into model
model.load_weights('model.h5')

# Perform real-time video emotion detection
settings = {
    'scaleFactor': 1.3, 
    'minNeighbors': 3, 
    'minSize': (50, 50), 
    'flags': 0
}

camera = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
count = 0

immatrix = np.empty((1, 1, 48,48), int)

while True:
	ret, img = camera.read()
	img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 

	detected = face_detector.detectMultiScale(img, **settings)

	# Make a copy as we don't want to draw on the original image:
	for x, y, w, h in detected:
		# Draw rectangle around region of interest -- the face
		cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)

		# Extract region of interest and convert to gray scale for processing
		maxLength = w+5
		if(h > w): maxLength = h+5
		roi = img[y:y+maxLength, x:x+maxLength]
		roi = cv2.cvtColor( roi, cv2.COLOR_RGB2GRAY )
		# We might need to resize the roi to 48x48 for the network
		roi = cv2.resize(roi, (48,48)) 
		cv2.imshow('ROI', roi)
		immatrix[0] = roi

		# Use the model we created to predict the emotion
		results = model.predict(immatrix)
		# print "Results: ", results

		max_index = -1
		max_value = -1

		for i in range(len(results[0])):
			if(max_value < results[0][i]):
				max_value = results[0][i]
				max_index = i

		emotion = ''
		if max_index == 0 : emotion = 'angry'
		if max_index == 1 : emotion = 'fear'
		if max_index == 2 : emotion = 'happy'
		if max_index == 3 : emotion = 'sad'
		if max_index == 4 : emotion = 'suprise'
		if max_index == 5 : emotion = 'neutral'

		# Report emotion on image
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img,emotion,(x,y+h+30), font, 1,(255,0,0),2,cv2.LINE_AA)

	cv2.imshow('facedetect', img)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyWindow("facedetect")





