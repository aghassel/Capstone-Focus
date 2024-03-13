import os
import pyaudio
import queue
from google.cloud import speech
from google.cloud import translate_v2 as translate
from ctypes import *
from contextlib import contextmanager

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

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'

class MicrophoneStream:
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True
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
        # with noalsaerr():
        #     self._audio_interface = pyaudio.PyAudio()
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
            yield b"".join(data)

def listen_print_loop(responses, translate_client):
    for response in responses:
        if not response.results:
            return
        result = response.results[0]
        if not result.alternatives:
            return
        transcript = result.alternatives[0].transcript
        # Check if the detected language is not English, then translate
        print(result.language_code)
        if result.language_code.lower() != "en-us" and result.language_code:
            translation = translate_client.translate(transcript, target_language='en')
            print(f"Original ({result.language_code}): {transcript}")
            print(f"Translated to English: {translation['translatedText']}")
        else:
            print(f"Detected English: {transcript}")

def main():
    rate = 16000  # Example rate
    # rate = 16000  # Example rate
    chunk = int(rate / 10)  # Example chunk size
    translate_client = translate.Client()

    client = speech.SpeechClient()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=rate,
        language_code='en-US',
        alternative_language_codes=['es-ES', 'fr-CA'],
        enable_automatic_punctuation=True
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    # stream = MicrophoneStream(rate, chunk)
    # stream.__enter__()

    while True:
        # audio_generator = stream.generator()
        with MicrophoneStream(rate, chunk) as stream:
            requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in stream.generator())
            responses = client.streaming_recognize(streaming_config, requests)
            # print(requests)
            # listen_print_loop(responses, translate_client)
            for response in responses:
                # if not response.results:
                #     continue
                # result = response.results[0]
                # if not result.alternatives:
                #     continue
                # if not result.is_final:
                #     continue
                if response.results:
                    result = response.results[0]
                    if result.alternatives:
                        transcript = result.alternatives[0].transcript
                        # Check if the detected language is not English, then translate
                        print(result.language_code)
                        if result.language_code.lower() != "en-us":
                            translation = translate_client.translate(transcript, target_language='en')
                            print(f"Original ({result.language_code}): {transcript}")
                            print(f"Translated to English: {translation['translatedText']}")
                        else:
                            print(f"Detected English: {transcript}")
                        break
                continue
                # responses = client.streaming_recognize(streaming_config, requests)

            # # *_, response = responses
            # if not next(responses).results:
            #     print("dropped")
            #     continue
            # result = next(responses).results[0]
            # if not result.alternatives:
            #     print("Dropped")
            #     continue
            # transcript = result.alternatives[0].transcript
            # # Check if the detected language is not English, then translate
            # if result.language_code != "en-US" and result.language_code:
            #     translation = translate_client.translate(transcript, target_language='en')
            #     print(f"Original ({result.language_code}): {transcript}")
            #     print(f"Translated to English: {translation['translatedText']}")
            # else:
            #     print(f"Detected English: {transcript}")

if __name__ == "__main__":
    main()