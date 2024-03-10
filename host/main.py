import socket
import cv2
import numpy as np
import struct

from asl import process_asl, init_asl

init_asl()

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
        # Read and process messages based on their type
        header = conn.recv(5)  # Read the message type and length
        if not header:
            print("No header received, closing connection.")
            break
        message_type, message_length = struct.unpack('<BI', header)
        
        message_data = b""
        while len(message_data) < message_length:
            packet = conn.recv(message_length - len(message_data))
            if not packet:
                break
            message_data += packet

        if message_type == 0:  # Mode message
            mode = message_data.decode('utf-8')
        elif message_type == 1:  # Image data
            frame_data = np.frombuffer(message_data, dtype=np.uint8)
            frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
            if frame is not None:
                if mode == "ASL":
                    label, confidence = process_asl(frame)
                elif mode == "CAR":
                    label, confidence = process_car(frame)
                
                # Ensure label and confidence are not None
                label = label if label is not None else "Unknown"
                confidence = confidence if confidence is not None else 0.0
                
                response = f"{label},{confidence:.2f}".encode('utf-8')
                conn.sendall(response)
            else:
                print("Failed to decode frame.")
finally:
    conn.close()
    server_socket.close()
