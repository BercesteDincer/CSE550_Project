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
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy 
import numpy as np
import os
from keras import backend as k
import tensorflow as tf

import numpy as np
import h5py

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


        model_file = msg.model_file
        weights_file = msg.weights_file

        #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        #os.environ["CUDA_VISIBLE_DEVICES"] = self.pid


        batch_size = 128
        num_classes = 10
        epochs = 12

        img_rows, img_cols = 28, 28

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        start = int(msg.start)
        end = int(msg.end)

        #start = 0
        #end = 5
        x_train = x_train[start:end, :]
        y_train = y_train[start:end]

        json_file = open(model_file, 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        # load weights into new model
        model.load_weights(weights_file)
        print("Loaded model from disk")

        old_weights = model.get_weights()

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.SGD(),
                      metrics=['accuracy'])

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=1,
                  verbose=1,
                 validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        new_weights = model.get_weights()

        # other_model = model
        # outputTensor = other_model.output 
        # listOfVariableTensors = other_model.trainable_weights
        # gradients = k.gradients(outputTensor, listOfVariableTensors)

        # trainingExample = x_train
        # sess = tf.InteractiveSession()
        # sess.run(tf.initialize_all_variables())

        # evaluated_gradients = sess.run(gradients,feed_dict={other_model.input:trainingExample})

        # weights = [tensor for tensor in model.trainable_weights]
        # optimizer = model.optimizer

        # gradients = optimizer.get_gradients(model.total_loss, weights)

        
        #print("GRADIENTS", evaluated_gradients[0])
       

        evaluated_gradients = []
        for i in range(len(new_weights)):
            evaluated_gradients.append(old_weights[i] - new_weights[i])
        print("EVALUATED GRADIENTS: ", len(evaluated_gradients))

        self.logged_values.append("Node {} completed training".format(self.pid))
        message = {"pid": self.pid, "gradients": evaluated_gradients}
        self.system.log_result(message)


            
                
