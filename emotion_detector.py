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

while True:
	ret, img = camera.read()
	img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 

	detected = face_detector.detectMultiScale(img, **settings)

	# Make a copy as we don't want to draw on the original image:
	for x, y, w, h in detected:
		# Draw rectangle around region of interest -- the face
		cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)

		# Extract region of interest and convert to gray scale for processing
		roi = img[y:y+h, x:x+w]
		roi = cv2.cvtColor( roi, cv2.COLOR_RGB2GRAY )
		# cv2.imshow('ROI', roi)
		# We might need to resize the roi to 48x48 for the network
		# img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 

		# Use the model we created to predict the emotion
		results = model.predict(roi)

		# Report emotion on image
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img,'Result',(x,y+h+30), font, 1,(255,0,0),2,cv2.LINE_AA)

	cv2.imshow('facedetect', img)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyWindow("facedetect")





