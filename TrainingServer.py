from collections import namedtuple, defaultdict
from multiprocessing import Process, Queue, JoinableQueue
import random
import time
import socket, threading
import queue
import numpy as np

from Messages import *
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
                dest, msg = self.funnel.get(timeout=0.5)
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

    #Method for logging the successful results
    def log_result(self, source, instance, value, status):
        self.accepted_results_q.put((source, instance, value, status))

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
        
    #Run forever to receive and send client messages 
    def run(self):

        while True:

            #Read from the successful request queue and send message to user
            while not system.accepted_results_q.empty():
                result  = system.accepted_results_q.get()
                clientSocket = client_dict[v["client_id"]]
                msg = result
                clientSocket.send(bytes(msg,'UTF-8'))
                

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
            
            #Read the data user provides
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            data_size = x_train.shape[0]
            print("DATASET TRAINING SIZE IS ", data_size)

            for i in range(number_of_nodes):
                start = i * (data_size / number_of_nodes)
                end = (i + 1) * (data_size / number_of_nodes)
                message = {"start": start, "end": end}
                system.messenger.send(-1, i, TrainingMsg(None, start, end))
                print("SENDING TO NODE", i, "START: ", start, "END: ", end)            
        
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

   
    #Keep receiving new client connections and creating new threads for them
    while True:
        server.listen(1)
        clientsock, clientAddress = server.accept()
        client_dict[clientAddress[1]] = clientsock
        newthread = ClientThread(clientAddress, clientsock)
        newthread.start()



     

       