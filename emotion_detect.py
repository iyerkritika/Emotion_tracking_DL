#!/bin/env python3
# SBATCH -N 2 # No. of computers you wanna use. Typically 1
# SBATCH -n 2 # No. of CPU cores you wanna use. Typically 1
# SBATCH -p gpu # This flag specifies that you wanna use GPU and not CPU
# SBATCH -o project.out # output file name, in case your program has anything to output (like print, etc)
# SBATCH -t 24:00:00 # Amount of time
# SBATCH --gres=gpu:3 # No. of GPU cores you wanna use. Usually 2-3

###########################################################
# Deep Learning for Advanced Robot Perception
# Project - Emotion detection
# Convolutional Neural Network for Emotion Recognition
# prepared by: Katie Gandomi, Kritika Iyer
###########################################################

##################### IMPORT DEPENDENCIES #################
import numpy as np
from numpy import *
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import LearningRateScheduler
from keras.layers import Activation
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from PIL import Image
import pickle
import glob

number_of_labels = 6
####################### METHODS #########################

def load_data():

	immatrix,label= pickle.load(open('database.pickle','rb'),encoding='latin1')
	print ("Number of input samples:", len(label))
	print ("images:",immatrix.shape)

	immatrix=np.reshape(immatrix,(36000,1,48,48))

	# Now Shuffle all the data and return the values
	data,Label = shuffle(immatrix,label, random_state=2)
	main_data = [data,Label]
	# # print (Label[0])
	# # Seperate main data out
	(X, Y) = (main_data[0], main_data[1])
	#
	# # Create the test and training data by splitting them
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=4)
	#
	# # convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(Y_train, number_of_labels)
	Y_test = np_utils.to_categorical(Y_test, number_of_labels)

	# Return the data
	return X_train, Y_train, X_test, Y_test

def create_model():
	# number_of_labels = 5

	# Create CNN model

	model = Sequential()
	model.add(Convolution2D(16, 3, 3, dim_ordering="th", input_shape=(1, 48, 48), activation='relu'))
	model.add(Convolution2D(16, 3, 3, dim_ordering="th"))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(32, 3, 3, dim_ordering="th", activation='relu'))
	model.add(Convolution2D(32, 3, 3, dim_ordering="th"))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(32, 3, 3, dim_ordering="th", activation='relu'))
	model.add(Convolution2D(32, 3, 3, dim_ordering="th"))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# model.add(Convolution2D(32, 3, 3, dim_ordering="th", activation='relu'))
	# model.add(Convolution2D(32, 3, 3, dim_ordering="th"))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(32, activation='relu'))
	# model.add(Dropout(0.1))
	model.add(Dense(number_of_labels, activation='softmax'))
	return model

def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.5
	epochs_drop = 8.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

def compile_model(model, X_train, Y_train, X_test, Y_test):
	# Set up parameters for compiling the model
	epochs = 20
	#lrate = 0.001
	#decay = lrate/epochs
	sgd = SGD(lr=0, momentum=0.9, decay=0, nesterov=False)
	adam = Adam(lr=0.0, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

	# Compile the model
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	print(model.summary())
	lrate = LearningRateScheduler(step_decay)
	callbacks_list = [lrate]

	# Fit the model
	history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=epochs, callbacks = callbacks_list, batch_size=64)

	# Final evaluation of the model
	scores = model.evaluate(X_test, Y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))

	return model, history, scores,model

#################################################################
#################### MAIN : CODE RUNS HERE ######################
#################################################################

X_train, Y_train, X_test, Y_test = load_data()
print ("Data Loaded Sucessfully!")

model = create_model()
print ("Model Created")
#
final_model, history, scores,model = compile_model(model, X_train, Y_train, X_test, Y_test)
print ("Model Compiled")

# Save Model data
model_json = final_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Model Saved to Disk Sucessfully!")

# f = open('output_history.pickle','wb')
# pickle.dump(history.history,f)
# f.close()
# f = open('output_scores.pickle','wb')
# pickle.dump(scores,f)
# f.close()
