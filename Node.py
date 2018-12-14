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


        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

        # Parameters
        learning_rate = 0.01
        training_epochs = 1
        batch_size = 100
        display_step = 1

        # Parameters
        learning_rate = 0.01
        training_epochs = 10
        batch_size = 100
        display_step = 1

        # tf Graph Input
        x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
        y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

        # Set model weights
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))

        # Construct model
        pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

        # Minimize error using cross entropy
        cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

        grad_W, grad_b = tf.gradients(xs=[W, b], ys=cost)


        new_W = W.assign(W - learning_rate * grad_W)
        new_b = b.assign(b - learning_rate * grad_b)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(mnist.train.num_examples/batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    # Fit training using batch data
                    _, _,  c = sess.run([new_W, new_b ,cost], feed_dict={x: batch_xs,
                                                               y: batch_ys})
                    
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if (epoch+1) % display_step == 0:
        #             print(sess.run(W))
                    print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

            print ("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy for 3000 examples
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print ("Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))
        
        evaluated_gradients = [grad_W, grad_b]
        self.logged_values.append("Node {} completed training".format(self.pid))
        message = {"pid": self.pid, "gradients": evaluated_gradients}
        self.system.log_result(message)


            
                
