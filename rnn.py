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
from keras.callbacks import ModelCheckpoint
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

# Function for applying temp to output
def apply_temp(z, temperature):
  z = np.array(z)**(1/temperature)
  q = z/z.sum()
  return np.argmax(q)

def data_division (fname, wsize, stride):
    global vocab_size
    print("Method 1: creates the training data to perform back propagation through time.")
    in_data = read_file(fname)
    #encode each character as a number
    chars = sorted(list(set(in_data)))
    mapping = dict((c, i) for i, c in enumerate(chars))
    
    encoded_line = np.array([mapping[char] for char in in_data])
    #encoded_line = in_data

    print("encoded_line shape and data:")
    print(encoded_line.shape)
    print(encoded_line)

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
    
    # x_list = np.array(x_list)
    # y_list = np.array(y_list)

    vocab_size = len(mapping)
    print("vocab_size:", vocab_size)
    encoded_x = []
    for x in x_list:
        #print(x)        
        one_hot_line = to_categorical(x, num_classes=vocab_size)
        #print(one_hot_line.shape)
        encoded_x.append(one_hot_line)
    
    encoded_x = np.array(encoded_x)
    print("encoded_x shape:", encoded_x.shape)
        
    encoded_y = []
    for y in y_list:
        one_hot_line = to_categorical(y, num_classes=vocab_size)
        encoded_y.append(one_hot_line)
    encoded_y = np.array(encoded_y)
    print("encoded_y shape:", encoded_y.shape)

    init_char = np.random.randint(0,vocab_size,vocab_size)

    return encoded_x, encoded_y,init_char
""" 
    # reshape X to be [samples, time steps, features]
    X = np.reshape(x_list, (len(x_list), wsize, 1))
    # normalize
    X = X / float(vocab_size)
    # one hot encode the output variable
    Y = to_categorical(y_list)
    
    return X, Y
 """

def char_prediction (init_char, model, temp, n,fname):
    init_char = init_char.tolist()

    """chars = sorted(list(set(init_char)))
    mapping = dict((c, i) for i, c in enumerate(chars))    
    encoded_line = np.array([mapping[char] for char in init_char])
    """

    in_data = read_file(fname)
    chars = sorted(list(set(in_data)))
    int_map = dict((i, c) for i, c in enumerate(chars))
    pred_seq=""
    print("Predication Start")
    for i in range(n):
        init_char = np.reshape(init_char, (1, 1, len(init_char)))
        # Predicting next character
        pred_num = model.predict(init_char, verbose=0)
        char_index = apply_temp(pred_num,temp)
        new_char = int_map[char_index]
        pred_seq = pred_seq + new_char
        # Progress the sequence
        init_char = np.delete(init_char,0)
        init_char = np.append(init_char,char_index)
    print("\n")
    print("Predicted Sequence (string):", pred_seq)
    print("seq length:", len(pred_seq))
    return pred_seq
    #write a method that predicts a given number of characters given a certain model and some characters to initialize


def model_train(model, y_train,x_train,epochs,lr,decay,hidden_state):
    #x_train = np.reshape(x_train,(np.newaxis,x_train.shape[0],x_train.shape[1]))
    #x_train = x_train[:,None]
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    rmodel = Sequential()
    if model == "lstm":
        #rmodel.add(layers.LSTM(100, input_shape=(1, x_train.shape[2])))
        rmodel.add(layers.LSTM(hidden_state, input_shape=(None, x_train.shape[2])))    #working

        #trying different model configurations


    elif model == "simple":
        rmodel.add(layers.SimpleRNN(hidden_state,return_sequences=True, input_shape=(None, x_train.shape[2])))
    rmodel.add(layers.Dropout(0.2))
    rmodel.add(layers.Dense(vocab_size, activation='softmax'))
        
    opt = keras.optimizers.Adam(learning_rate=lr,decay=decay)
    rmodel.compile(loss='mean_squared_error', optimizer=opt)    
        
        # define the checkpoint
    #filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    filepath="weights/weights{epoch:03d}.hdf5"
    #checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, mode='min',save_freq = 5)
        # fit the model
    #rmodel.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=2,callbacks=[checkpoint])
    history = rmodel.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=2,callbacks=[checkpoint])
    rmodel.summary()
    #rmodel.add(layers.Dense(1))
    """ print("From training -- vocab_size:", vocab_size)
    rmodel.add(layers.Dense(vocab_size, activation='softmax'))
    opt = keras.optimizers.Adam(learning_rate=lr,decay=decay)
    rmodel.compile(loss='mean_squared_error', optimizer=opt)
    #rmodel.compile(loss='categorical_crossentropy', optimizer='adam')
    #checkpoint = keras.callbacks.ModelCheckpoint("model{epoch:08d}", period=20)
    
    filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    rmodel.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=2,callbacks=[checkpoint])
    #rmodel.fit(x_train, y_train, epochs=epochs, batch_size=2, callbacks=callbacks_list)
    #rmodel.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=2)
    # print(rmodel.callbacks) """
    return rmodel,history



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

    x_train, y_train,init_char = data_division(fname, window_size, stride)
    print("Before training: x_train[0]:", x_train[0])
    n=20
    lr = 0.001
    decay = 0.0
    name = model + str(hidden_state) + str(window_size) + str(stride) + str(temp)
    rmodel,history = model_train(model,y_train,x_train,20,lr,decay,hidden_state)
    plt.plot(history.history["loss"])
    plt.title("Loss for " + name)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig(name + ".png")
    plt.clf()
    x_train = np.reshape(x_train,(1,x_train.shape[0],x_train.shape[1],x_train.shape[2]))
    
    # Printing predictions every certain number of epochs to see output
    filepath = "weights/weights{epoch:03d}.hdf5"
    filepaths = [filepath.format(epoch = x) for x in np.arange(1,21,5)]
    for filepath in filepaths:
        rmodel.load_weights(filepath)
        opt = keras.optimizers.Adam(learning_rate=lr,decay=decay)
        rmodel.compile(loss='mean_squared_error', optimizer=opt)
        char_prediction(init_char,rmodel,temp,n,fname)

    # Printing for last epoch
    filepath = "weights/weights020.hdf5"    #the file name of the best weights
    rmodel.load_weights(filepath)
    opt = keras.optimizers.Adam(learning_rate=lr,decay=decay)
    rmodel.compile(loss='mean_squared_error', optimizer=opt)

    pred_seq = char_prediction(init_char,rmodel,temp,n,fname)
    
    f = open(name + ".txt", "w")
    f.write(pred_seq)
    f.close()
    