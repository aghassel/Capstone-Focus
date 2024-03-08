import io
import socket
import struct
import time
import picamera

import RPi.GPIO as GPIO

button_pin = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

server_ip = '0.0.0.0'
server_port = 8000
client_socket = socket.socket()
client_socket.connect((server_ip, server_port))
connection = client_socket.makefile('wb')

current_mode = "ASL"

def button_callback(channel):
    global current_mode
    if current_mode == "ASL":
        current_mode = "CAR"
    else:
        current_mode = "ASL"
    print(f"Mode changed to: {current_mode}")

GPIO.add_event_detect(button_pin, GPIO.FALLING, callback=button_callback, bouncetime=300)

try:
    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        camera.framerate = 24
        time.sleep(2)
        
        stream = io.BytesIO()
        for _ in camera.capture_continuous(stream, 'jpeg', use_video_port=True):
            connection.write(current_mode.encode('utf-8'))
            connection.flush()
            
            connection.write(struct.pack('<L', stream.tell()))
            connection.flush()
            
            stream.seek(0)
            connection.write(stream.read())
            
            stream.seek(0)
            stream.truncate()
            
            prediction = client_socket.recv(1024).decode('utf-8')
            print(f"Received prediction: {prediction}")

finally:
    connection.write(struct.pack('<L', 0))
    connection.close()
    client_socket.close()
    GPIO.cleanup()
