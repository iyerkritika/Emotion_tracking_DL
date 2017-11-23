###########################################################
# Deep Learning for Advanced Robot Perception
# Project - Categorize a single image for emotion
# Convolutional Neural Network for Emotion Recognition
# prepared by: Katie Gandomi, Kritika Iyer
###########################################################

# Import dependencies
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.models import model_from_json
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import cv2
import numpy as np

# load json and create model
json_file = open('../model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into model
model.load_weights('../model.h5')

# Choose Image to categorize
input_img = cv2.imread('categorize_images/neutral.png',0)
immatrix = np.empty((1, 1, 48,48), int)
immatrix[0] = input_img

# Predict results of the model
results = model.predict(immatrix)[0]*100 # Confidence as a percent
print results

# Emotion Level
emotion_labels = ('angry/disgust', 'fear', 'happy', 'sad', 'suprised', 'neutral')
y_pos = np.arange(len(emotion_labels))
print y_pos

# Plot Image and Confidence of Categorization
plt.figure(1)
plt.subplot(211)
plt.imshow(input_img, cmap='gray')
plt.subplot(212)
plt.bar(y_pos, results, align='center', alpha=0.5)
plt.xticks(y_pos, emotion_labels)
plt.ylim([0,100])
plt.xlabel('Emotions')
plt.ylabel('Confidence')
plt.show()


