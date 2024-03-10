import io
import socket
import struct
from picamera2 import Picamera2
import asyncio
from display import display_asl_translation

async def perform_asl_translation(picam2):
    server_ip = '192.168.2.14'
    server_port = 8000
    
    # Create a non-blocking socket
    client_socket = socket.socket()
    await asyncio.get_event_loop().sock_connect(client_socket, (server_ip, server_port))
    
    # Picamera2 setup is already provided in the argument
    camera_config = picam2.create_still_configuration(main={"size": (640, 480)})
    picam2.configure(camera_config)
    picam2.start()
    
    current_mode = "ASL"
    
    try:
        while True:
            picam2.capture_file("temp.jpg")  # Temporarily capture to a file
            with open("temp.jpg", "rb") as image_file:
                image_data = image_file.read()
                mode_message = current_mode.encode('utf-8')
                mode_header = struct.pack('<BI', 0, len(mode_message))  # Message type 0, length of mode message
                image_header = struct.pack('<BI', 1, len(image_data))  # Message type 1, length of image data
                
                # Send mode message
                await asyncio.get_event_loop().sock_sendall(client_socket, mode_header + mode_message)
                
                # Send image data
                await asyncio.get_event_loop().sock_sendall(client_socket, image_header + image_data)
                
                prediction = await asyncio.get_event_loop().sock_recv(client_socket, 1024)
                prediction_text = prediction.decode('utf-8')
                # Split the prediction to extract the word part only
                word, _ = prediction_text.split(',', 1)  # This assumes the format "word, confidence"
                print(f"Received prediction: {prediction_text}")
                
                # Display only the word part of the ASL translation
                display_asl_translation(word.strip())  # 
        
    finally:
        client_socket.close()
        picam2.stop()