###########################################################
# Deep Learning for Advanced Robot Perception
# Project - Emotion detector
# Confusion Matrix Generator
# prepared by: Katie Gandomi, Kritika Iyer
###########################################################

# Import dependencies
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.utils import shuffle
import numpy as np
import pickle

# load json and create model
json_file = open('../model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into model
model.load_weights('../model.h5')
print "Model Imported !"

# Load data
immatrix,label= pickle.load(open('database.pickle','rb'))
print ("Number of input samples:", len(label))
print ("images:",immatrix.shape)

immatrix=np.reshape(immatrix,(36000,1,48,48))

# Now Shuffle all the data and return the values
data,Label = shuffle(immatrix,label, random_state=2)
main_data = [data,Label]
	
# # Seperate main data out
(X, Y) = (main_data[0], main_data[1])
print "Data Loaded !"

# # Create the test and training data by splitting them
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=4)
# #
# # # convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(Y_train, number_of_labels)
# Y_test = np_utils.to_categorical(Y_test, number_of_labels)

# prediction and true labels
y_prob = model.predict(X, batch_size=32, verbose=0)
y_pred = [np.argmax(prob) for prob in y_prob]
y_true = [true for true in Y]
print "Prediction and True Labels Created"
print len(y_prob)
print y_prob[0]
print len(y_pred)
print y_pred[0]
print len(y_true)
print y_true[0]

emotion_labels = ['angry/disgust', 'fear', 'happy', 'sad', 'suprised', 'neutral']

# Create the confusion matrix
print "Creating Confusion"
cm = confusion_matrix(y_true, y_pred)
print cm.shape
print cm
cmap=plt.cm.YlGnBu
fig = plt.figure(figsize=(6,6))
matplotlib.rcParams.update({'font.size': 16})
ax  = fig.add_subplot(111)
matrix = ax.imshow(cm, interpolation='nearest', cmap=cmap)
fig.colorbar(matrix) 
print "Prediction results"
for i in range(0,6):
    for j in range(0,6):  
        ax.text(j,i,cm[i,j],va='center', ha='center')
        print cm[i,j]
# ax.set_title('Confusion Matrix')
ticks = np.arange(len(emotion_labels))
ax.set_xticks(ticks)
ax.set_xticklabels(emotion_labels, rotation=45)
ax.set_yticks(ticks)
ax.set_yticklabels(emotion_labels)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
print "Done!"





