import socket
import cv2
import numpy as np
import struct, io
from car_detection.car_detector import ObjectDetector

from asl import process_asl, init_asl

init_asl()

detector = ObjectDetector()

# def process_car(frame, threshold):
#     bbox = detector.predict_single_frame(frame, threshold)
#     return bbox

HOST = ''
PORT = 8000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print(f"Listening on port {PORT}...")

while True:
    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")
    run_server = True
    try:
        while run_server:
            # Read and process messages based on their type
            header = conn.recv(5)  # Read the message type and length
            # if addrfrom != addr:
            #     print(addrfrom, addr)
            #     continue
            if not header :
                print("No header received, closing connection.")
                # continue
                run_server = False
                conn.close()
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
                        conn.sendall(response)
                    elif mode == "CAR":
                        # response = ''

                        warning = detector.warning(frame)
                        # print("sending")
                        conn.sendto(warning.encode('utf-8'), addr)

        
                else:
                    print("Failed to decode frame.")
    finally:
        pass
    # conn.close()
    # server_socket.close()
# conn.close()
# server_socket.close()
