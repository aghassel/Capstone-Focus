import asyncio
import websockets
import socket
import struct
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from weather import weather
from schedule import get_schedule
from server import perform_asl_translation
from display import *
import time

# Initialize the PiCamera
picam = Picamera2()
recording = False

current_mode = 0
current_event_index = 0  # For schedule


async def take_picture():      # Take pic
    camera_config = picam.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
    picam.configure(camera_config)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"/home/focus/image_{timestamp}.jpg"
    picam.start()
    picam.capture_file(filename)
    print(f"Picture saved as {filename}")
    display_text(f"Picture \ntaken!")
    picam.stop()
    await asyncio.sleep(2)
    await display_mode_1_message()  


async def record_video(start=True):     # Take video
    global recording
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    if start:
        video_config = picam.create_video_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
        picam.configure(video_config)
        encoder = H264Encoder(bitrate=10000000)
        output = f"/home/focus/video_{timestamp}.h264"
        picam.start_recording(encoder, output)
        recording = True
        print("Recording started")
        display_text("Recording...\n\nHold to \nstop")
    else:
        picam.stop_recording()
        recording = False
        print("Recording stopped")
        display_text("Recording \nstopped\n\nVideo saved!")
        await asyncio.sleep(4) 
        await display_mode_1_message() 


async def display_weather_info():
    weather_info = await weather()
    print(weather_info)
    display_weather(weather_info)


def display_schedule_info(event_index=0):
    schedule_info = get_schedule()
    print(schedule_info)
    display_schedule(schedule_info, event_index)


async def display_mode_1_message():
    display_text("Tap to\ntake pic \n\n Hold to \nstart  video")


async def execute_mode_action():
    global current_mode
    if current_mode == 1:
        await display_mode_1_message()
    elif current_mode == 2:  # Weather
        await display_weather_info()
    elif current_mode == 3:  # Schedule
        display_schedule_info(current_event_index)
    elif current_mode == 4:  # ASL Translation
        await perform_asl_translation(picam)


async def listen_for_messages(uri):
    global current_mode, current_event_index
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            print(f"Message received: {message}")
            if message == "double":
                current_mode = (current_mode % 4) + 1  
                current_event_index = 0  # Reset the event index on mode change
                print(f"Mode changed to {current_mode}")
                await execute_mode_action()
            elif message == "single" and current_mode == 3:
                display_schedule_info(current_event_index)
                current_event_index += 1
            elif message == "long" and current_mode == 1:  # Video
                await record_video(start=not recording)
            elif message == "single" and current_mode ==1: # Pic
                await take_picture()


# Display a startup logo
#logo_path = '/path/to/your/logo.png'  # Update the path to your logo
#display_logo(logo_path)

# WebSocket URI
uri = "ws://192.168.2.53:8422"
asyncio.get_event_loop().run_until_complete(listen_for_messages(uri))
