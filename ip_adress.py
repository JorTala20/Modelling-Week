import socket

hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)
print(f"Local IP Address: {local_ip}")