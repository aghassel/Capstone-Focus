import io
import socket
import struct
from picamera2 import Picamera2
import asyncio
from display import *
from PIL import Image
import websockets

class asl_connection():
    def __init__(self):
        self.run = False
        self.server_ip = '192.168.2.74'
        self.server_port = 8000
        self.current_mode = "ASL"
        print("init")

    def startup(self, picam2):
        print("startup")
        self.picam2 = picam2
        # Create a non-blocking socket
        self.client_socket = socket.socket()    
        # Picamera2 setup is already provided in the argument
        self.camera_config = self.picam2.create_still_configuration(main={"size": (640, 480)})
        self.picam2.configure(self.camera_config)
        self.picam2.start()
    
    def set_run(self, mode):
        print("run: ", mode)
        self.run = mode
    
    async def running_loop(self, picam2):
        print("rinning")
        self.startup(picam2)
        print("startup complete")
        
        self.client_socket.connect((self.server_ip, self.server_port))
        
        # await asyncio.get_event_loop().sock_connect(self.client_socket, (self.server_ip, self.server_port))
        try:
            while self.run:
                self.picam2.capture_file("temp.jpg")  # Temporarily capture to a file
                with Image.open("temp.jpg") as img:
                    rotated_img = img.rotate(90, expand=True)  # `expand=True` to resize the image to fit the new orientation
                    rotated_img.save("temp.jpg")  # Overwrite the original image with the rotated one

                with open("temp.jpg", "rb") as image_file:
                    image_data = image_file.read()
                    mode_message = self.current_mode.encode('utf-8')
                    mode_header = struct.pack('<BI', 0, len(mode_message))  # Message type 0, length of mode message
                    image_header = struct.pack('<BI', 1, len(image_data))  # Message type 1, length of image data
                    
                    # Send mode message
                    # await asyncio.get_event_loop().sock_sendall(self.client_socket, mode_header + mode_message)
                    self.client_socket.sendall(mode_header + mode_message)
                    
                    # Send image data
                    # await asyncio.get_event_loop().sock_sendall(self.client_socket, image_header + image_data)
                    self.client_socket.sendall(image_header + image_data)
                    
                    # prediction = await asyncio.get_event_loop().sock_recv(client_socket, 1024)
                    prediction = self.client_socket.recv(1024)
                    if self.current_mode == "ASL":
                        prediction_text = prediction.decode('utf-8')
                        # Split the prediction to extract the word part only
                        word, _ = prediction_text.split(',', 1)  # This assumes the format "word, confidence"
                        print(f"Received prediction: {prediction_text}")
                        
                        # Display only the word part of the ASL translation
                        display_text(word.strip())  # 
                        await asyncio.sleep(0)
                    if self.current_mode == "CAR":
                        prediction_text = prediction.decode('utf-8')
                        # Split the prediction to extract the word part only
                        print(f"Received prediction: {prediction_text}")
                        
                        # Display only the word part of the ASL translation
                        display_text(prediction_text) 
                        # image = Image.frombytes('RGBA', prediction)
                        # display_img(image)
                        # image.save("testst.jpg")
                        await asyncio.sleep(0)
            self.client_socket.close()
            self.picam2.stop()    
            
        finally:
            self.client_socket.close()
            self.picam2.stop()
    
    def close_server(self):
        self.set_run(False)
        self.client_socket.close()
        self.picam2.stop()


class server_connection():
    def __init__(self):
        self.run = False
        self.server_ip = '192.168.2.74'
        self.server_port = 8000
        self.current_mode = "ASL"
        print("init")

    def startup(self, picam2):
        print("startup")
        self.picam2 = picam2
        # Create a non-blocking socket
        self.client_socket = socket.socket()    
        # Picamera2 setup is already provided in the argument
        self.camera_config = self.picam2.create_still_configuration(main={"size": (640, 480)})
        self.picam2.configure(self.camera_config)
        self.picam2.start()
    
    def set_run(self, mode):
        print("run: ", mode)
        self.run = mode
    
    def running_loop(self, picam2):
        print("rinning")
        self.startup(picam2)
        print("startup complete")
        
        self.client_socket.connect((self.server_ip, self.server_port))
        
        # await asyncio.get_event_loop().sock_connect(self.client_socket, (self.server_ip, self.server_port))
        try:
            self.picam2.capture_file("temp.jpg")  # Temporarily capture to a file
            with Image.open("temp.jpg") as img:
                rotated_img = img.rotate(90, expand=True)  # `expand=True` to resize the image to fit the new orientation
                rotated_img.save("temp.jpg")  # Overwrite the original image with the rotated one

            with open("temp.jpg", "rb") as image_file:
                image_data = image_file.read()
                mode_message = self.current_mode.encode('utf-8')
                mode_header = struct.pack('<BI', 0, len(mode_message))  # Message type 0, length of mode message
                image_header = struct.pack('<BI', 1, len(image_data))  # Message type 1, length of image data
                
                # Send mode message
                # await asyncio.get_event_loop().sock_sendall(self.client_socket, mode_header + mode_message)
                self.client_socket.sendall(mode_header + mode_message)
                
                # Send image data
                # await asyncio.get_event_loop().sock_sendall(self.client_socket, image_header + image_data)
                self.client_socket.sendall(image_header + image_data)
                
                # prediction = await asyncio.get_event_loop().sock_recv(client_socket, 1024)
                prediction = self.client_socket.recv(1024)
                if self.current_mode == "ASL":
                    prediction_text = prediction.decode('utf-8')
                    # Split the prediction to extract the word part only
                    word, _ = prediction_text.split(',', 1)  # This assumes the format "word, confidence"
                    print(f"Received prediction: {prediction_text}")
                    
                    # Display only the word part of the ASL translation
                    display_text(word.strip())  # 
                if self.current_mode == "CAR":
                    prediction_text = prediction.decode('utf-8')
                    # Split the prediction to extract the word part only
                    print(f"Received prediction: {prediction_text}")
                    
                    # Display only the word part of the ASL translation
                    display_text(prediction_text) 
            
        finally:
            self.client_socket.close()
            self.picam2.stop()


async def perform_asl_translation(picam2):
    server_ip = '172.20.10.3'
    server_port = 8000
    
    # Create a non-blocking socket
    client_socket = socket.socket()
    await asyncio.get_event_loop().sock_connect(client_socket, (server_ip, server_port))
    # client_socket.connect((server_ip, server_port))
    
    # Picamera2 setup is already provided in the argument
    camera_config = picam2.create_still_configuration(main={"size": (640, 480)})
    picam2.configure(camera_config)
    picam2.start()
    
    current_mode = "ASL"
    
    try:
        while True:
            picam2.capture_file("temp.jpg")  # Temporarily capture to a file
            with Image.open("temp.jpg") as img:
                rotated_img = img.rotate(90, expand=True)  # `expand=True` to resize the image to fit the new orientation
                rotated_img.save("temp.jpg")  # Overwrite the original image with the rotated one

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
                if current_mode == "ASL":
                    prediction_text = prediction.decode('utf-8')
                    # Split the prediction to extract the word part only
                    word, _ = prediction_text.split(',', 1)  # This assumes the format "word, confidence"
                    print(f"Received prediction: {prediction_text}")
                    
                    # Display only the word part of the ASL translation
                    display_asl_translation(word.strip())  # 
                if current_mode == "CAR":
                    image = Image.frombytes('RGBA', prediction)
                    display_img(image)
                    image.save("testst.jpg")
                await asyncio.sleep(0)
        
    finally:
        client_socket.close()
        picam2.stop()


# async def main():
#     asl_conn = asl_connection()
#     asl_conn.set_run(True)
#     await asl_conn.running_loop(Picamera2())

# if __name__ == "__main__":
#     asyncio.run(main())
    