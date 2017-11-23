###########################################################
# Deep Learning for Advanced Robot Perception
# Project - Emotion detector
# CNN Layer Visualization
# prepared by: Katie Gandomi, Kritika Iyer
###########################################################
# Import dependencies
import theano
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.models import model_from_json
from keras import backend as K
import os
import cv2
import numpy as np

def plot_interlayer_outputs(model, input_img, layer_num1, layer_num2, colormaps=False):
    output_fn = K.function([model.layers[layer_num1].input, K.learning_phase()],[model.layers[layer_num2].output, K.learning_phase()])
    im = output_fn([input_img, 0])[0] # test mode
    print im.shape

    n_filters = im.shape[1]
    fig = plt.figure(figsize=(12,6))
    for i in range(n_filters):
        ax = fig.add_subplot(8,8,i+1)
        if colormaps:
            ax.imshow(im[0,i,:,:], cmap='Blues')#seq_colors[i]
        else:
            ax.imshow(im[0,i,:,:], cmap=matplotlib.cm.gray) #matplotlib.cm.gray
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    plt.show()

# Use Theano backend
os.environ['KERAS_BACKEND'] = "theano"
K.set_learning_phase(0)
reload(K)

# load json and create model
json_file = open('../model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into model
model.load_weights('../model.h5')

# Visualize layers
input_img = cv2.imread('layer_image.png',0)
immatrix = np.empty((1, 1, 48,48), int)
immatrix[0] = input_img

# plot_interlayer_outputs(model, immatrix, 0, 1, colormaps=True)
plot_interlayer_outputs(model, immatrix, 0, 2, colormaps=True)
plot_interlayer_outputs(model, immatrix, 0, 3, colormaps=True)
plot_interlayer_outputs(model, immatrix, 0, 4, colormaps=True) 
plot_interlayer_outputs(model, immatrix, 0, 10, colormaps=True)




