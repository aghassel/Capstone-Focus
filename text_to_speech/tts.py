import configparser
import os
from openai import OpenAI
import pyaudio
from fastapi import FastAPI
import threading

class TTS:
    def __init__(self, name=None, verbose=False, api_key=None):
        '''
        Name options: alloy, echo, fable, onyx, nova, and shimmer
        Reads the API key from the api_keys.ini file in the same directory or from the environment
        '''
        print ("TTS __init__")

        # read the .ini file from the directory
        if api_key is not None:
            self.api_key = api_key
        else:
            self.config = configparser.ConfigParser()
            self.config.read(os.path.join(os.path.dirname(__file__), 'api_keys.ini'))
            self.api_key = os.environ.get('OPENAI_API_KEY') or self.config.get('openai', 'OPENAI_API_KEY')  # get the API key from the environment or the .ini file
        self.client = OpenAI(api_key=self.api_key)
        self.name = "nova"
        self.app = FastAPI()
        self.verbose = verbose
        self.hist = ""

        if name is not None:
            self.name = name

    def generate_tts(self, text):
        '''
        Generate the TTS audio from the given text
        returns the response object
        '''
        return self.client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice=self.name,
            input=text,
            response_format="wav"
        )
        

    def generate_and_play_tts(self, text):
        '''
        Generate the TTS audio from the given text and play it
        Example use:
        tts = TTS()
        tts.generate_and_play_tts("Hello, world!")
        NOTE: Does not run asynchronously
        '''
        try:
            tic = time.time()
            response = self.generate_tts(text)
            # play the audio
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=22050,
                            output=True)
            with response as res:
                if res.status_code == 200:
                    for chunk in res.iter_bytes(chunk_size=2048):
                        stream.write(chunk)
            toc = time.time()
            if self.verbose:
                print(f"Time taken for audio playback: {toc - tic:.2f} seconds")
            stream.stop_stream()
            stream.close()
            p.terminate()

        except Exception as e:
            print(f"Error: {e}")


    def generate_and_play_tts_async(self, text):
        '''
        Generate the TTS audio from the given text and play it asynchronously
        Example use:
        tts = TTS()
        tts.generate_and_play_tts("Hello, world!")
        '''
        # Create a new thread that runs the _generate_and_play_tts method
        if self.hist != text or text.lower() != "unknown":
            self.hist = text
            print (self.hist)

            threading.Thread(target=self._generate_and_play_tts_async, args=(text,)).start()
        else:
            print ("Text is the same as the last one, not playing")      

    def _generate_and_play_tts_async(self, text):
        '''
        This is the method that actually generates and plays the TTS audio.
        It's called by the generate_and_play_tts method in a new thread.
        '''
        try:
            tic = time.time()
            response = self.generate_tts(text)
            # play the audio
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=22050,
                            output=True)
            with response as res:
                if res.status_code == 200:
                    for chunk in res.iter_bytes(chunk_size=2048):
                        stream.write(chunk)
            toc = time.time()
            if self.verbose:
                print(f"Time taken for audio playback: {toc - tic:.2f} seconds")
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            print(f"Error: {e}")
            

def main(args):
    text = args.text

    tts = TTS('nova', verbose=True)

    print ('Testing Sync')
    tts.generate_and_play_tts(text)
 
    print ('Testing Async')
    text = "Hello, world!"

    tts.generate_and_play_tts_async(text)
    for i in range(20):
        print(i)
        time.sleep(0.5)
    tts.generate_and_play_tts_async(text)

    tts.generate_and_play_tts_async("unknown")



    print ("done") 
    print ('Test to see if audio plays while waiting for the loop to finish')


if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="I wonder what abdellah had for lunch today, hopefully there was no dairy in his meal. He's been having a lot of stomach issues lately. Oh, poor abdellah. He's being so brave.")
    args = parser.parse_args()
    main(args)

