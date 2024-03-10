from luma.core.interface.serial import spi
from luma.oled.device import ssd1306
from luma.core.render import canvas
from PIL import Image, ImageFont, ImageDraw
import RPi.GPIO as GPIO
import time
from datetime import datetime

# OLED display width and height setup
WIDTH = 128
HEIGHT = 64

# GPIO setup for reset pin
RST_PIN = 23

# SPI connection parameters
SPI_PORT = 0
SPI_DEVICE = 0
SPI_BUS_SPEED_HZ = 8000000  # Adjust as necessary for your display

# Setup GPIO for RST
GPIO.setwarnings(False)  # Disable GPIO warnings
GPIO.setmode(GPIO.BCM)
GPIO.setup(RST_PIN, GPIO.OUT)

# Reset the display
GPIO.output(RST_PIN, GPIO.LOW)  # Pull RST low to reset
time.sleep(0.1)  # Wait 100ms
GPIO.output(RST_PIN, GPIO.HIGH)  # Bring RST back high

# Initialize SPI serial connection and the OLED display
serial_spi = spi(port=SPI_PORT, device=SPI_DEVICE, gpio_DC=24, gpio_RST=RST_PIN, bus_speed_hz=SPI_BUS_SPEED_HZ, gpio_CS=8)
device = ssd1306(serial_spi, width=WIDTH, height=HEIGHT, rotate=1) 


def display_text(text):
    with canvas(device) as draw:
        draw.text((2, 25), text, fill="white")


def draw_multiline_text(draw, text, position, fill="white", default_font_height=10):
    """
    Draw text, automatically breaking into lines that fit within the screen width.
    Uses 'textlength' for measuring text width and calculates height based on the default font size.
    """
    max_width = device.width - position[0]
    words = text.split()
    current_line = ""
    x, y = position
    line_count = 1

    for word in words:
        test_line = f"{current_line} {word}" if current_line else word
        width = draw.textlength(test_line)
        if width < max_width:
            current_line = test_line
        else:
            # Draw the current line and start a new one
            draw.text((x, y), current_line, fill=fill)
            current_line = word
            y += default_font_height 
            line_count += 1
    
    if current_line:
        draw.text((x, y), current_line, fill=fill)


def display_weather(weather):
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    with canvas(device) as draw:
        draw.text((0, 0), current_time, fill="white")
        draw.text((0, 25), 'Weather:', fill="white")
        draw_multiline_text(draw, weather, (0, 40), fill="white")


def display_schedule(schedule, event_index):
    now = datetime.now()
    current_time = now.strftime("%H:%M")

    if event_index < len(schedule):
        event = schedule[event_index]
        with canvas(device) as draw:
            draw.text((0, 0), current_time, fill="white")
            draw_multiline_text(draw, event, (0, 30), fill="white", default_font_height=10)
    else:
        with canvas(device) as draw:
            draw.text((0, 0), "No more \nevents.", fill="white")


def display_asl_translation(translation):
    with canvas(device) as draw:
        # Display a header or title if desired
        draw.text((0, 0), "ASL \nTranslation:", fill="white")
        # Utilize the draw_multiline_text function to display the translation
        draw_multiline_text(draw, translation, (0, 35), fill="white", default_font_height=10)


def display_logo(path):
    with canvas(device) as draw:
        logo = Image.open(path).convert("1")
        logo = logo.resize((WIDTH, HEIGHT))  # Adjust size if necessary
        logo = logo.rotate(-90, expand=True)  # Rotate for portrait
        draw.bitmap((0, 0), logo, fill="white")