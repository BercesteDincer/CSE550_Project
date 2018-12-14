import socket
import time

#Define the client
SERVER = "127.0.0.1"
PORT = 8080
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((SERVER, PORT))
in_data =  client.recv(1024)
print("From Server :" ,in_data.decode())
 
#Continue receiving messages until the client is inactive
failed = 0
while True:
  print("Enter command:")
  out_data = input()
  start_time = time.time()
  client.sendall(bytes(out_data,'UTF-8'))
  if out_data=='bye':
    break
  in_data =  client.recv(1024)
  print("From Server :" ,in_data.decode())
  elapsed_time = time.time() - start_time

  print("TOOK ", elapsed_time)
client.close()
