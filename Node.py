import sys
import time
from threading import Thread
from multiprocessing import Queue
from queue import Empty

from Message import *

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

#Basic class for a node in system
class Node:
  
    def __init__(self, pid, system, messenger):
        
        self.pid = pid #pid of the node
        self.system = system #keep a reference to the system to get the ids of other nodes
        self.active = True #decide whether to stop the node
        self.messenger = messenger #messenger for sending messages

        self.logged_values = [] #keep a list of values logged by paxos
       
        
    #Method for running a node forever
    def run(self):
        
        print("Node {} started".format(self.pid))
        #continue receiving while node is alive
        while self.active:
            msg = self.recv() #receive the message
            self.handle_message(msg) #handle the message 
        print("Node {} shutting down".format(self.pid))

    #Method for sending message from the node to another node
    def send_message(self, msg, pids):
        for pid in pids:
            print("Node {} sending message to {} (message = {})".format(self.pid, pid, msg))
            self.messenger.send(self.pid, pid, msg)

    #Method for receiving message from other nodes
    def recv(self):
        msg = self.messenger.recv(self.pid)
        source = getattr(msg, 'source', None)
        print("Node {} received message from {} (message = {})".format(self.pid, source, msg))
        return msg

    def message_done(self):
        self.messenger.task_done(self.pid)

    #Method for deciding on actions for different message types
    def handle_message(self, msg):

        ######### SYSTEM QUIT MESSAGE ###########
        if msg == 'quit':
            self.handle_quit()
            
        if isinstance(msg, TrainingMsg):
            self.train(msg)

    #Method for quitting the node
    def handle_quit(self):
        self.active = False

    #Method for training
    def train(self, msg):

        import os
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = self.pid


        batch_size = 128
        num_classes = 10
        epochs = 12

        img_rows, img_cols = 28, 28

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        start = int(msg.start)
        end = int(msg.end)

        x_train = x_train[start:end, :]
        y_train = y_train[start:end]

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
            
                