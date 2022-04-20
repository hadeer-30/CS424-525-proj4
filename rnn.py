# -*- coding: utf-8 -*-
"""
    Authors : Hadeer Farahat and Cayse Rogers
    Purpose : Project4 for SP22-CS424/525 class

"""

from fileinput import filename
import os
#from pyexpat import model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
from PIL import Image
import glob

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp


from sklearn.preprocessing import OneHotEncoder
import numpy as np


import sys

# Function for sampling output
def sample(z, temperature):
  z = np.array(z)**(1/temperature)
  q = z/z.sum()
  return np.argmax(np.random.multinomial(1, q, 1))

def data_division (fname, wsize, stride):
    print("Method 1: creates the training data to perform back propagation through time.")
    in_data = read_file(fname)
    #encode each character as a number
    char_to_int = dict((c, i) for i, c in enumerate(set(in_data)))

    #char_to_int = (ord(c) for c in (in_data))


    #pp ...
    # le = pp.LabelEncoder()
    # char_to_int = in_data.apply(le.fit_transform)
    print( "ch to i:", char_to_int)
    
    #encode text to integers
    data = [char_to_int[char] for char in in_data]
    print("int_text: ", data)

       

    data_len = len(data)
    print("data length:", data_len)
    print("window size:", wsize)
    print("stride:", stride)
    #break the data into multiple sequences of length wsize+1, with a moving window of size stride
    data_list = []
    for i in range(0, data_len - wsize, stride):
        data_list.append(data[i:i+wsize+1])
        #data_list.append(in_data[i:i+wsize+1])
    print("data list:", data_list)
    print("length of data list:", len(data_list))

    """ divide the data into x and y sequences. x would be the input sequence you are using (of size wsize) and y will be the output sequence (also of size wsize but starting a character later) """
    x_list = []
    y_list = []

    for i in range(len(data_list)):
        x_list.append(data_list[i][:-1])
        y_list.append(data_list[i][1:])
    print("x_list:", x_list)
    print("y_list:", y_list)

    """ one hot encode the data. That is x and y should each be of size (num of sequences * wsize * vocab size) """

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(x_list)
    x_list = enc.transform(x_list).toarray()
    print("x_list after transform:", x_list)
    print("x_list shape:", x_list.shape)

    enc.fit(y_list)
    y_list = enc.transform(y_list).toarray()
    print("y_list after transform:", y_list)
    print("y_list shape:", y_list.shape)

    #checking results
    convert = enc.inverse_transform(y_list)
    print("y_list after inverse transform:", convert)
    
    #convert to numpy array
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    print("x_list shape:", x_list.shape)
    print("y_list shape:", y_list.shape)

    return x_list, y_list


def char_prediction (init_char, model, temp, n):
    print(model.predict(init_char))
    #write a method that predicts a given number of characters given a certain model and some characters to initialize


def model_train(model, y_train,x_train,epochs,lr,decay):
    #x_train = np.reshape(x_train,(np.newaxis,x_train.shape[0],x_train.shape[1]))
    #x_train = x_train[:,None]
    rmodel = Sequential()
    if model == "lstm":
        rmodel.add(layers.LSTM(1, input_shape=x_train.shape))
    elif model == "simple":
        rmodel.add(layers.SimpleRNN(1, input_shape=x_train.shape))
    rmodel.add(layers.Dense(1))
    opt = keras.optimizers.Adam(learning_rate=lr,decay=decay)
    rmodel.compile(loss='mean_squared_error', optimizer=opt)
    checkpoint = keras.callbacks.ModelCheckpoint("model{epoch:08d}", period=20)
    rmodel.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=2,callbacks=[checkpoint])
    print(rmodel.callbacks)
    return rmodel



#Read text file information
def read_file (fname):
    #print("External function: reads the text file.")
    with open(fname, 'r') as f:
        text = f.read()
    return text


if __name__=="__main__":
    """ 
    From project4 write-up:
    python3 rnn.py beatles.txt lstm 100 10 5 1
    Will run the code with an LSTM with a hidden state of size 100 and a window size of 10, a stride of 5 and a sampling temperature of 1.
    """
    fname = sys.argv[1]
    #print("fname:", fname)
    model = sys.argv[2]
    #print("model:", model)
    hidden_state = int(sys.argv[3])
    #print("hidden_state:", hidden_state)
    window_size = int(sys.argv[4])
    #print("window_size:", window_size)
    stride = int(sys.argv[5])
    #print("stride:", stride)
    temp = int(sys.argv[6])
    #print("temp:", temp)

    x_train, y_train = data_division(fname, window_size, stride)
    n=5
    rmodel = model_train(model,y_train,x_train,100,0.5,0)
    char_prediction(np.array([ord('c')]),rmodel,temp,n)
    

