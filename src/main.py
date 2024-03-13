import asyncio
import signal
import threading
import pyaudio
import wave
import numpy as np
import websockets
import socket
import struct
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from PIL import Image
from weather import weather
from schedule import get_schedule
from server import perform_asl_translation
from speech2text import speech_to_text
from question import question_mode
from wake_words import detect_wake_word, transcribe_audio, record_until_pause
from display import *
import time
import dropbox

dropbox_access_token = 'sl.BxOMpl5J2xXTdDSFJzbWOoCRAcv9GWuz-ySLjcOPMj7x5eF8f7TD5IeDIDqAEZqrXNVPMzExXqV5J-0F9PAAXQ1P3xYfFwGD9vNQNgvQQ29bx6Jo3kc7Lq53dovR8Ox0rPUnuH6IeaU2'

# Initialize the PiCamera
picam = Picamera2()
recording = False

current_mode = 0
current_event_index = 0  # For schedule


async def take_picture():  # Take pic, flip it, then save
    camera_config = picam.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
    picam.configure(camera_config)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"/home/focus/image_{timestamp}.jpg"
    picam.start()
    picam.capture_file(filename)
    print(f"Picture saved as {filename}")
    img = Image.open(filename)
    img_rotated = img.rotate(90) 
    img_rotated.save(filename)
    print(f"Rotated picture saved as {filename}")
    display_text("Picture taken and rotated!")
    
    dropbox_path= "/Focus_SmartGlasses/{}".format(filename)   #Go to your Dropbox profile and make a folder named SmartGlassesAPI
    client = dropbox.Dropbox(dropbox_access_token)
    client.files_upload(open(filename, "rb").read(), dropbox_path)
    print('image saved to dropbox account')

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
    display_text("Tap to take pic \n\n Hold to start  video")


async def execute_mode_action():
    global current_mode
    if current_mode == 1:
        await display_mode_1_message()
    elif current_mode == 2:  # Weather
        await display_weather_info()
    elif current_mode == 3:  # Schedule
        display_schedule_info(current_event_index)
    elif current_mode == 4:  # Speech to Text
        await speech_to_text()
    elif current_mode == 5:  # ASL translation
        await perform_asl_translation(picam)
    elif current_mode == 6:
        display_text("Question and answering...")
        record_until_pause()
        question = transcribe_audio("output.wav")
        print(question)
        answer = await question_mode(question)
        print(answer)
        display_text(answer)


async def listen_for_messages(uri):
    global current_mode, current_event_index
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            print(f"Message received: {message}")
            if message == "double":
                current_mode = (current_mode % 5) + 1  
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


async def listen_for_wake_word_and_commands():
    while True:
        # Detect wake word
        await asyncio.get_event_loop().run_in_executor(None, detect_wake_word)
        print("Wake word detected! Listening for command...")
        display_text("Listening...")

        # Record command after wake word detection
        filename = await asyncio.get_event_loop().run_in_executor(None, record_until_pause)
        print(f"filename: {filename}")

        # Transcribe audio to text
        transcription = await asyncio.get_event_loop().run_in_executor(None, transcribe_audio, filename)
        
        if transcription:
            # Process transcription to execute the appropriate action
            transcription = transcription.replace(".", "").replace(",", "").replace("?", "").replace("!", "")
            transcription = transcription.replace(":", "").replace(";", "").lower()
            print(transcription)

            # Process command
            if "picture" in transcription:
                current_mode = 1
                await take_picture()
            elif "video" in transcription:
                current_mode = 1
                await record_video(start=True)  # Assuming you want to start recording immediately
            # Add more elif blocks for other commands like "weather", "schedule", etc.
            elif "weather" in transcription:
                current_mode = 2
                await display_weather_info()
            elif "schedule" in transcription:
                current_mode = 3
                display_schedule_info(current_event_index)
            elif "transcribe" in transcription:
                current_mode = 4
                display_text("Transcribing...")
                await speech_to_text()
            elif "sign language" in transcription:
                current_mode = 5
                await perform_asl_translation(picam)
            elif "question" in transcription:
                current_mode = 6
                display_text("Question and answering...\n\n\nWhat is your question?")
                record_until_pause()
                question = transcribe_audio("output.wav")
                print(question)
                answer = await question_mode(question)
                print(answer)
                display_text(answer)
                #tts(answer)
            else:
                display_text("Unknown command. Please try again.")


# Display a startup logo
#logo_path = '/path/to/your/logo.png'  # Update the path to your logo
#display_logo(logo_path)

# WebSocket URI
uri = "ws://0.0.0.0:8422"

async def main():
    # Set up your asyncio tasks here
    task1 = asyncio.create_task(listen_for_messages(uri))
    task2 = asyncio.create_task(listen_for_wake_word_and_commands())

    # Function to handle cancellation so tasks can be cancelled gracefully
    def signal_handler():
        task1.cancel()
        task2.cancel()

    # Registering signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        # Instead of waiting for both tasks to complete, enter an infinite loop
        while True:
            # Sleep for a long time (effectively forever) without blocking
            await asyncio.sleep(3600)  # Sleep for 1 hour
    except asyncio.CancelledError:
        # If the loop or tasks are cancelled, ensure everything is stopped gracefully
        pass
    finally:
        # Wait for tasks to be cancelled
        await task1
        await task2

# Running the asyncio event loop
asyncio.run(main())