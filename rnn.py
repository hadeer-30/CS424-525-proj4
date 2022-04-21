# -*- coding: utf-8 -*-
"""
    Authors : Hadeer Farahat and Cayse Rogers
    Purpose : Project4 for SP22-CS424/525 class

"""

from fileinput import filename
import os
from re import X
#from pyexpat import model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
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

vocab_size = 0

# Function for sampling output
def sample(z, temperature):
  z = np.array(z)**(1/temperature)
  q = z/z.sum()
  return np.argmax(np.random.multinomial(1, q, 1))

def data_division (fname, wsize, stride):
    global vocab_size
    print("Method 1: creates the training data to perform back propagation through time.")
    in_data = read_file(fname)
    #encode each character as a number
    chars = sorted(list(set(in_data)))
    mapping = dict((c, i) for i, c in enumerate(chars))
    encoded_line = np.array([mapping[char] for char in in_data])
    # print("encoded_line shape and data:")
    # print(encoded_line.shape)
    # print(encoded_line)

    data_len = len(encoded_line)
    data_list = []
    for i in range(0, data_len - wsize, stride):
        data_list.append(encoded_line[i:i+wsize+1])
    
    data_list = np.array(data_list)
    # print("data list:", data_list)
    # print("length of data list:", len(data_list))

    
    #break the data into multiple sequences of length wsize+1, with a moving window of size stride    
    """ divide the data into x and y sequences. x would be the input sequence you are using (of size wsize) and y will be the output sequence (also of size wsize but starting a character later) """
    x_list = []
    y_list = []

    for i in range(len(data_list)):
        x_list.append(data_list[i][:-1])
        y_list.append(data_list[i][1:])
    
    x_list = np.array(x_list)
    y_list = np.array(y_list)

    vocab_size = len(mapping)
    encoded_x = []
    for x in x_list:
        print(x)        
        one_hot_line = to_categorical(x, num_classes=vocab_size)
        print(one_hot_line.shape)
        encoded_x.append(one_hot_line)
    
    encoded_x = np.array(encoded_x)
    print("encoded_x shape:", encoded_x.shape)
        
    encoded_y = []
    for y in y_list:
        one_hot_line = to_categorical(y, num_classes=vocab_size)
        encoded_y.append(one_hot_line)
    encoded_y = np.array(encoded_y)
    print("encoded_y shape:", encoded_y.shape)

    return encoded_x, encoded_y


def char_prediction (init_char, model, temp, n):
    print(model.predict(init_char))
    #write a method that predicts a given number of characters given a certain model and some characters to initialize


def model_train(model, y_train,x_train,epochs,lr,decay):
    #x_train = np.reshape(x_train,(np.newaxis,x_train.shape[0],x_train.shape[1]))
    #x_train = x_train[:,None]
    print("x_train shape:", x_train.shape)
    rmodel = Sequential()
    if model == "lstm":
        rmodel.add(layers.LSTM(1, input_shape=(1, x_train.shape[2])))

    elif model == "simple":
        rmodel.add(layers.SimpleRNN(1,stateful=True,batch_size=1, return_sequences=True, input_shape=(None, x_train.shape[2])))
    rmodel.add(layers.Dense(1))
    print("From training -- vocab_size:", vocab_size)
    rmodel.add(layers.Dense(vocab_size, activation='softmax'))
    #rmodel.add(layers.Dense(1))
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
    