import socket
import cv2
import numpy as np
import struct
from car_detection.car_detector import CarDetector


detector = CarDetector()


def process_car(frame):
    bbox = detector.predict_single_frame(frame, threshold=0.5)
    return bbox

from asl import process_asl, init_asl

init_asl()



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
                    # Ensure label and confidence are not None
                    label = label if label is not None else "Unknown"
                    confidence = confidence if confidence is not None else 0.0
                    response = f"{label},{confidence:.2f}".encode('utf-8')
                elif mode == "CAR":
                    bbox = process_car(frame)
                    response += f"{len(bbox)}".encode('utf-8')
                    for i in bbox:
                        response += f",{i[0]},{i[1]},{i[2]},{i[3]}".encode('utf-8')   
                    print (response)               
         
                conn.sendall(response)
            else:
                print("Failed to decode frame.")
finally:
    conn.close()
    server_socket.close()
