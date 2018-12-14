from collections import namedtuple, defaultdict
from multiprocessing import Process, Queue, JoinableQueue
import random
import time
import socket, threading
import queue
import numpy as np
import h5py

from Message import *
from Node import *


#Messenger class for sending and receiving messages across nodes
class Messenger:
    
    def __init__(self, system, num_nodes):
    

        self.num_nodes = num_nodes
        self.system = system
        self.funnel = Queue()
        self.inbox = [Queue() for i in range(self.num_nodes)]
        self.message_count = 0

        self.active = True
        self.terminate = False

    #Run forever to receive and send messages
    def run(self):
        while True:
            if not self.active and self.terminate:
                break
            try:
                dest, msg = self.funnel.get(timeout=500)
            except queue.Empty:
                pass
            else:
                self.inbox[dest].put(msg)

    #Method for sending message
    def send(self, from_, to, msg):
        self.message_count += 1
        self.funnel.put((to, msg))


    #Method for receiving message
    def recv(self, from_):
        return self.inbox[from_].get()

    def task_done(self, pid):
        self.funnel.task_done()



#Class for defining the Paxos system consisting of nodes
class TrainingServer():
   
    #Create the default system
    def __init__(self, num_nodes):

        #Queues for log values
        self.accepted_results_q = Queue()
        self.failed_results_q = Queue()

        #set the ids of agents in the system
        self.num_nodes = num_nodes
        self.node_ids = list(range(0, num_nodes))
        self.message_timeout= 3

        #start messenger threads
        self.messenger = Messenger(self, self.num_nodes)
        self.messenger_thread = Thread(target=self.messenger.run)
        self.messenger_thread.start()

        #start all node processes
        self.processes = self.launch_processes()

        self.model_file = ''
        self.weights_file = ''
        self.input_file = ''

    #Method for logging the successful results
    def log_result(self, source):
        self.accepted_results_q.put((source))

    #Method for logging the failed results
    def log_failure(self, source, value, status):
        self.failed_results_q.put((source, value, status))

    #Method for starting all processes
    def launch_processes(self) :
        processes = []
        for pid in range(self.num_nodes):
            node = Node(pid, self, self.messenger)
            p = Process(target=node.run)
            p.start()
            processes.append(p)
        return processes

    #Join all node processes
    def join(self):
        for process in self.processes:
            process.join()



#Thread for managing the communication between lock server and clients
class CommunicationThread(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self)
        self.node_responses = np.zeros(number_of_nodes)
        self.node_gradients = []
        self.current_epoch = 0
        self.epoch_count = 5
        self.remaining_nodes = np.arange(number_of_nodes)
        
    #Run forever to receive and send client messages 
    def run(self):

        last_active = 0
        while self.current_epoch < self.epoch_count:


            #Keep flag for last received data
            
            #Make sure all nodes responded
            if not system.accepted_results_q.empty():
                result  = system.accepted_results_q.get()
                #print("RESULT ", result)
                self.node_responses[int(result['pid'])] = 1
                self.node_gradients.append(result['gradients'])
                last_active = time.time()
                print('Last active updated, ', last_active)  

            current_time = time.time()
            if( not np.all(self.node_responses == 1) ):
                time.sleep(2)
                current_time = time.time()
                #print('Dont worry',  ((current_time - last_active)))


            current_time = time.time()
            if( (not np.all(self.node_responses == 1) ) and (last_active != 0) and (current_time - last_active) > 10):
                flag = 0
                print("WE COULD NOT RECEIVE ALL RESPONSES...")
                #Detect which nodes did not finish
                failed_nodes = np.where(self.node_responses == 0)[0][0]
                print("FAILED NODES ARE: ", failed_nodes)
                self.failed_nodes = failed_nodes
                self.node_responses[:] = 0
                self.node_gradients = []
                self.node_responses[failed_nodes] = 1

                #Read the data user provides
                if system.input_file == 'mnist':
                    (x_train, y_train), (x_test, y_test) = mnist.load_data()
                
                data_size = x_train.shape[0]
                print("DATASET TRAINING SIZE IS ", data_size)

                self.remaining_nodes = []
                for i in range(number_of_nodes):
                    if i != failed_nodes:
                        self.remaining_nodes.append(i)

                # remaining_start = failed_nodes * (data_size / number_of_nodes)
                # remaining_end = (failed_nodes + 1) * (data_size / number_of_nodes)
                # for i in range(len(self.remaining_nodes)):
                #     start = remaining_start  + (i ) * ((remaining_end - remaining_start)) / len(self.remaining_nodes)
                #     end =  remaining_start + (i + 1) * ((remaining_end - remaining_start)) / len(self.remaining_nodes)
                #     message = {"start": start, "end": end}
                #     if self.current_epoch == 0:
                #         system.messenger.send(-1, self.remaining_nodes[i], TrainingMsg(None, start, end, system.model_file, system.weights_file, system.input_file))
                #         print("SENDING TO NODE", self.remaining_nodes[i], "START: ", start, "END: ", end, " MODEL FILE: ", system.model_file, " WEIGHTS FILE ", system.weights_file)            
                #     else:
                #         system.messenger.send(-1, self.remaining_nodes[i], TrainingMsg(None, start, end, system.model_file, new_weights_file, system.input_file))
                #         print("SENDING TO NODE", self.remaining_nodes[i], "START: ", start, "END: ", end, " MODEL FILE: ", system.model_file, " WEIGHTS FILE ", new_weights_file)            
                
                if system.input_file == 'mnist':
                    #Read the data user provides
                    (x_train, y_train), (x_test, y_test) = mnist.load_data()
            
                data_size = x_train.shape[0]
                print("DATASET TRAINING SIZE IS ", data_size)
                
                for i in range(len(self.remaining_nodes)):
                        start = i * (data_size / len(self.remaining_nodes))
                        end = (i + 1) * (data_size / len(self.remaining_nodes))
                        message = {"start": start, "end": end}
                        if self.current_epoch == 0:
                            system.messenger.send(-1, self.remaining_nodes[i], TrainingMsg(None, start, end, system.model_file, system.weights_file, system.input_file))
                            print("SENDING TO NODE", self.remaining_nodes[i], "START: ", start, "END: ", end, " MODEL FILE: ", system.model_file, " WEIGHTS FILE ", system.weights_file)            
                        else:
                            system.messenger.send(-1, self.remaining_nodes[i], TrainingMsg(None, start, end, system.model_file, new_weights_file, system.input_file))
                            print("SENDING TO NODE", self.remaining_nodes[i], "START: ", start, "END: ", end, " MODEL FILE: ", system.model_file, " WEIGHTS FILE ", new_weights_file)            
                    


            #When all results are received
            if np.all(self.node_responses == 1):
                print("ALL RESULTS COLLECTED FOR EPOCH ", self.current_epoch)

                json_file = open(system.model_file, 'r')
                model_json = json_file.read()
                json_file.close()
                model = model_from_json(model_json)
                # load weights into new model
                

                #Now update weights
                #Read all weights  
                if self.current_epoch == 0:
                    model.load_weights(system.weights_file)
                    print("Loaded model from disk")
                    new_weights = model.get_weights()
                else:
                    model.load_weights( system.weights_file[:-3] + '_update' + str(self.current_epoch - 1))
                    print("Loaded model from disk")
                    new_weights = model.get_weights()


                print("REMAINING NODE ", self.remaining_nodes)
                print("EPOCH ", self.current_epoch)
                for i in range(len(new_weights)):
                    updates = np.zeros(new_weights[i].shape)

                    for n in range(len(self.remaining_nodes)):
                        updates = updates + (self.node_gradients[n][i])
                        self.node_responses[self.remaining_nodes[n]] = 0
                    new_weights[i] =  new_weights[i] - (updates / number_of_nodes)
                
                self.node_gradients = []

                #Update the weights
                updated_weights = new_weights
                
                model.set_weights(updated_weights)
                print("Updated weights")
                
                new_weights_file = system.weights_file[:-3] + '_update' + str(self.current_epoch)
                model.save_weights(new_weights_file)
                print("Updated_model")

                self.current_epoch = self.current_epoch + 1

                if self.current_epoch >= self.epoch_count:

                    num_classes = 10

                    print("!!!!!!!!!!!!!!TRAINING COMPLETED!!!!!!!!")
                    # the data, split between train and test sets
                    if system.input_file == 'mnist':
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

                    json_file = open(system.model_file, 'r')
                    model_json = json_file.read()
                    json_file.close()
                    model = model_from_json(model_json)
                    # load weights into new model
                    model.load_weights(new_weights_file)
                    print("Loaded model from disk: ", new_weights_file)

                    model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

                    score = model.evaluate(x_test, y_test, verbose=0)
                    print('Test loss:', score[0])
                    print('Test accuracy:', score[1])

                    print("Training is completed...")  
                    clientSocket = clientsock
                    msg = "MODEL SUCCESSFULLY TRAINED AND SAVED, FINAL ACCURACY IS " + str(score[1])
                    clientSocket.send(bytes(msg,'UTF-8'))

                else:
                    #Read the data user provides
                    if system.input_file == 'mnist':
                        (x_train, y_train), (x_test, y_test) = mnist.load_data()
                    
                    data_size = x_train.shape[0]
                    print("DATASET TRAINING SIZE IS ", data_size)

                    if len(self.remaining_nodes) != number_of_nodes:
                        print("One node failed...")
                        for i in range(len(self.remaining_nodes)):
                            start = i * (data_size / len(self.remaining_nodes))
                            end = (i + 1) * (data_size / len(self.remaining_nodes))
                            message = {"start": start, "end": end}
                            system.messenger.send(-1, self.remaining_nodes[i], TrainingMsg(None, start, end, system.model_file, new_weights_file, system.input_file))
                            print("SENDING TO NODE", self.remaining_nodes[i], "START: ", start, "END: ", end, " MODEL FILE: ", system.model_file, " WEIGHTS FILE ", new_weights_file)            


                    else:
                        for i in range(number_of_nodes):
                            start = i * (data_size / number_of_nodes)
                            end = (i + 1) * (data_size / number_of_nodes)
                            message = {"start": start, "end": end}
                            system.messenger.send(-1, i, TrainingMsg(None, start, end, system.model_file, new_weights_file, system.input_file))
                            print("SENDING TO NODE", i, "START: ", start, "END: ", end, " MODEL FILE: ", system.model_file, " WEIGHTS FILE ", new_weights_file)            

            #ALL TRAINING DONE
        
class NodeFailureThread(threading.Thread):
    
    def __init__(self, messenger):
        threading.Thread.__init__(self)
        self.messenger = messenger
        
    #Run forever to receive failed nodes from server
    def run(self):
        while True:
            out_data = input()
            tokens = out_data.split(" ")
            if len(tokens) == 2 and tokens[0] == 'quit':
                print("Node {} is stopped".format(tokens[1]))
                self.messenger.send(-1, int(tokens[1]), "quit")



#Creating a new thread for handling client requests independently
class ClientThread(threading.Thread):

    #Initiate new client threas
    def __init__(self,clientAddress,clientsocket):
        threading.Thread.__init__(self)
        self.csocket = clientsocket
        self.clientAddress = clientAddress
        print ("New connection added: ", self.clientAddress)
    
    #Receive and send client requests
    def run(self):

        print ("Connection from : ", self.clientAddress)
        self.csocket.send(bytes("Hi, you are client " + str(self.clientAddress),'utf-8'))
        
        msg = ''
        
        #Keep receiving messages until client decides to quit
        while True:
            data = self.csocket.recv(2048)
            msg = data.decode()
            if msg=='bye' or msg == '':
                break
            #Read the model and weights file from the user
            model_file, weights_file, input_file = msg.split(' ')

            system.model_file = model_file
            system.weights_file = weights_file
            system.input_file = input_file

            if system.input_file == 'mnist':
                #Read the data user provides
                (x_train, y_train), (x_test, y_test) = mnist.load_data()
            
            data_size = x_train.shape[0]
            print("DATASET TRAINING SIZE IS ", data_size)

            
            for i in range(number_of_nodes):
                start = i * (data_size / number_of_nodes)
                end = (i + 1) * (data_size / number_of_nodes)
                message = {"start": start, "end": end}
                system.messenger.send(-1, i, TrainingMsg(None, start, end, model_file, weights_file, input_file))
                print("SENDING TO NODE", i, "START: ", start, "END: ", end, " MODEL FILE: ", model_file, " WEIGHTS FILE ", weights_file)            


        print("ALL NODES JOINED...")

        print ("Client at ", self.clientAddress, " disconnected...")   

import sys

if __name__ == "__main__":

    number_of_nodes = 5

    #Define lock server ports and start system
    LOCALHOST = "127.0.0.1"
    PORT = 8080
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((LOCALHOST, PORT))
    
    system = TrainingServer(number_of_nodes)

    print("Deep Learning Training Server is started")

    #keep track of all results sent to user
    instance_results_sent_log = set()
        
    #keep a dictionary of all clients
    client_dict = {}

    #Create thread for handling client requests
    print("Waiting for client requests...")
    communicationThread = CommunicationThread()
    communicationThread.start()
    failureThread = NodeFailureThread(system.messenger)
    failureThread.start()

   
    #Keep receiving new client connections and creating new threads for them
    while True:
        server.listen(1)
        clientsock, clientAddress = server.accept()
        client_dict[clientAddress[1]] = clientsock
        newthread = ClientThread(clientAddress, clientsock)
        newthread.start()



     

  
