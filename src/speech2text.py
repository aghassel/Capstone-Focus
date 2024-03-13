import asyncio
import os
import queue
import pyaudio
from google.cloud import speech
from google.cloud import translate_v2 as translate
from ctypes import *
from contextlib import contextmanager
from display import display_speech2text, display_text

# Suppress ALSA error messages
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

def py_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)

# Set the path to the Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'

class MicrophoneStream:
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True
        with noalsaerr():
            self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
            input_device_index=0,
        )
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                break
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        break
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b''.join(data)


async def async_listen_print_loop(responses, translate_client):
    for response in responses:
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue
        transcript = result.alternatives[0].transcript
        print(f"Detected: {transcript}")
        display_text(transcript)
        # Implement any additional functionality you need, such as translation
        await asyncio.sleep(0)  # Yield control back to the event loop

async def speech_to_text():
    rate = 44100  # Audio sample rate
    chunk = int(rate / 10)  # Chunk size for processing
    translate_client = translate.Client()  # Initialize translation client

    client = speech.SpeechClient()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=rate,
        language_code='en-US',  # Specify the initial language code
        enable_automatic_punctuation=True,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    while True:
        with MicrophoneStream(rate, chunk) as stream:
            audio_generator = stream.generator()
            def geny():
                for content in audio_generator:
                    # asyncio.sleep(0).__await__()
                    yield speech.StreamingRecognizeRequest(audio_content=content)
            # requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)
            requests = geny()
            responses = client.streaming_recognize(streaming_config, requests)
            # await async_listen_print_loop(responses, translate_client)
            for response in responses:
                if not response.results:
                    continue
                result = response.results[0]
                if not result.alternatives:
                    continue
                transcript = result.alternatives[0].transcript
                print(f"Detected: {transcript}")
                if transcript.lower().find("hey focus") != -1:
                    return
                display_text(transcript)
                await asyncio.sleep(0.2)  # Yield control back to the event loop
            # Implement any additional functionality you need, such as translation
            



    # with MicrophoneStream(rate, chunk) as stream:
    #         requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in stream.generator())
    #         responses = client.streaming_recognize(streaming_config, requests)
    #         for response in responses:
    #             if response.results:
    #                 result = response.results[0]
    #                 if result.alternatives and result.is_final:
    #                     transcript = result.alternatives[0].transcript
    #                     # Check if the detected language is not English, then translate
    #                     if result.language_code != "en-US" and result.language_code:
    #                         translation = translate_client.translate(transcript, target_language='en')
    #                         print(f"Original ({result.language_code}): {transcript}")
    #                         print(f"Translated to English: {translation['translatedText']}")
    #                         display_text(translation['translatedText'])
    #                     else:
    #                         print(f"Detected English: {transcript}")
    #                     break
    #             continue



