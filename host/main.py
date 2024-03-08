import socket
import cv2
import numpy as np
import struct


from asl import process_asl, init_asl

init_asl()

# placeholder for liam
def process_car(frame):
    return "Car Detected", 0.95

HOST = ''  
PORT = 8000  

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print(f"Listening on port {PORT}...")

conn, addr = server_socket.accept()
print(f"Connected by {addr}")

try:
    while True:
        mode = conn.recv(3).decode('utf-8')  
        
        data = conn.recv(4)
        frame_size = struct.unpack('<L', data)[0]
        
        frame_data = b""
        while len(frame_data) < frame_size:
            packet = conn.recv(4096)
            if not packet: break
            frame_data += packet
        
        frame = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        
        if mode == "ASL":
            label, confidence = process_asl(frame)
        elif mode == "CAR":
            label, confidence = process_car(frame)
        else:
            continue  
        
        response = f"{label},{confidence:.2f}".encode('utf-8')
        conn.sendall(response)
        
finally:
    conn.close()
    server_socket.close()
    print("Connection closed.")
