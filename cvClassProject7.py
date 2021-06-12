#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 03:49:33 2021

@author: rysul
"""
#import dependencies
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.utils import np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import regularizers, optimizers
import numpy as np
from matplotlib import pyplot
import cv2 as cv

def buildModel():
    # build the model
    
    # number of hidden units variable 
    # we are declaring this variable here and use it in our CONV layers to make it easier to update from one place
    base_hidden_units = 64 # changed from 32 to 64
    
    # l2 regularization hyperparameter
    weight_decay = 1e-4 
    num_classes = 10
    
    # instantiate an empty sequential model 
    model = Sequential()
    
    # CONV1
    # notice that we defined the input_shape here because this is the first CONV layer. 
    # we don’t need to do that for the remaining layers
    model.add(Conv2D(base_hidden_units, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    # CONV2
    model.add(Conv2D(base_hidden_units, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    # CONV3
    model.add(Conv2D(2*base_hidden_units, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    # CONV4
    model.add(Conv2D(2*base_hidden_units, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    # CONV5
    model.add(Conv2D(4*base_hidden_units, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    # CONV6
    model.add(Conv2D(4*base_hidden_units, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    # FC7
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    
    # print model summary
    model.summary()
    model.load_weights('model.125epochs.hln64bs256.hdf5')

    return model




def createPyramid(image, levels):
    '''
    creates an image pyramid for given number of levels
    '''
    gaussPyr = [image]
    
    for i in range(levels-1):
        image = cv.pyrDown(image)
        gaussPyr.append(image)
        
    return gaussPyr



def classification(model, mean, std, labelNames):
    '''
    for classifying images captured by my webcame
    takes in the model and label names. Also mean and standard deviation of the input image
    '''
    #for capturing video using the camera
    cap = cv.VideoCapture(0)
    imgCounter = 0
    cv.namedWindow("Prediction")
    # to check if the camera opened successfully
    if (cap.isOpened()== False):
      print("Error opening video stream or file")
    
    while(cap.isOpened()):
        code, img = cap.read()
        if code == True:
            k = cv.waitKey(1)
            cv.imshow("Prediction", img)
            if k%256 == ord('q'):
                print(k)
                # q pressed for quitting
                print("Quitting predictions.")
                break
            elif k%256 == ord('c'):
                print(k)
                # c pressed for capturing
                print("Capturing webcam frame for predictions.")
                imgName = "opencv_frame_{}.png".format(imgCounter)
                cv.imwrite('capturedImages/'+imgName, img)
                print("{} written!".format(imgName))
                print("Shape of the original imgae: {}".format(img.shape))
                
                # processing the captured image
                # bgr to rgb conversion
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                # resize the image to 256x256
                imgResized = cv.resize(img, (256,256), interpolation = cv.INTER_AREA)
                print("Shape of the resized image: {}".format(imgResized.shape))
                pyramid = createPyramid(imgResized, 4) # calling 4 levels
                print("Shape of the lowest pyramid image: {}".format(pyramid[3].shape))
                cv.imwrite('output/'+imgName.split('.')[0]+'pyr.jpg', pyramid[3])
                
                # prediction
                # adding a new axis on pyramid[3]
                evalImage = pyramid[3][np.newaxis,:,:,:]
                evalImage = (evalImage-mean)/(std+1e-7)
                print("Shape of the image being evaluated {}".format(evalImage.shape))
                pred = model.predict(evalImage)
                # predictions in probabilites
                np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
                print("The probablities: {}".format(pred))
                # predicted class
                print("Predicted class: {}\n".format(labelNames[pred.argmax()]))
                print(labelNames[pred.argmax()])
                
                imgCounter += 1
        else:
            print("failed to grab frame")
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':    
    
    # building the model
    cnnModel = buildModel()
    # exporting the model
    #tf.keras.utils.plot_model(cnnModel, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)
    
    # CIFAR-10 labels
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # mean and standard deviation of the training data
    mean = 120.70748
    std =  64.150024
    classification(cnnModel, mean, std, labels)
    