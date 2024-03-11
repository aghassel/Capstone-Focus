import configparser
import os
from openai import OpenAI
import pyaudio
from fastapi import FastAPI



class TTS:
    def __init__(self, name=None):
        '''
        Name options: alloy, echo, fable, onyx, nova, and shimmer
        '''
        print ("TTS __init__")

        # read the .ini file from the directory
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(os.path.dirname(__file__), 'api_keys.ini'))
        self.api_key = os.environ.get('OPENAI_API_KEY') or self.config.get('openai', 'OPENAI_API_KEY')  # get the API key from the environment or the .ini file
        self.client = OpenAI(api_key=self.api_key)
        self.name = "nova"
        self.app = FastAPI()

        if name is not None:
            self.name = name

    def generate_tts(self, text):
        return self.client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice=self.name,
            input=text,
            response_format="wav"
        )

def main(args):
    tts = TTS("nova")
    
    tic = time.time()
    response = tts.generate_tts(args.text)
    toc = time.time()
    print(f"Time taken for request: {toc - tic:.2f} seconds")
    # play the audio
    tic = time.time()
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=22050,
                    output=True)
    with response as res:
        if res.status_code == 200:
            for chunk in res.iter_bytes(chunk_size=2048):
                stream.write(chunk)
    stream.stop_stream()
    stream.close()
    p.terminate()
    toc = time.time()
    print(f"Time taken for audio playback: {toc - tic:.2f} seconds")
    print("done")




if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="Hello, world! This is a test. Abdellah Go Shower!")
    args = parser.parse_args()
    main(args)

