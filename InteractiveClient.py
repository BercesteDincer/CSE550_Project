
import socket

#Define the client
SERVER = "127.0.0.1"
PORT = 8080
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((SERVER, PORT))
client.settimeout(3)
in_data =  client.recv(1024)
print("From Server :" ,in_data.decode())
 
#Continue receiving messages until the client is inactive
failed = 0
while True:
  print("Enter command:")
  out_data = input()
  client.sendall(bytes(out_data,'UTF-8'))
  if out_data=='bye':
    break
  try:
  	in_data =  client.recv(1024)
  except socket.timeout:
  	continue
  print("From Server :" ,in_data.decode())
client.close()


