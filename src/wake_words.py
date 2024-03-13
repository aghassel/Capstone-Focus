import pvporcupine
import pyaudio
import numpy as np
import struct
import os
import wave
import openai
from openai import OpenAI
import dotenv
dotenv.load_dotenv()


def detect_wake_word():
    pa = pyaudio.PyAudio()
    porcupine = pvporcupine.create(
        access_key=os.environ.get("PORCUPINE_API_KEY"),
        keyword_paths=['wake_words/Hey-Focus_en_raspberry-pi_v3_0_0.ppn', 'wake_words/Okay-Focus_en_raspberry-pi_v3_0_0.ppn']
    )
    audio_stream = pa.open(rate=porcupine.sample_rate, channels=1,
                           format=pyaudio.paInt16, input=True,
                           frames_per_buffer=porcupine.frame_length)

    print("Listening for the wake word...")

    while True:
        pcm = audio_stream.read(porcupine.frame_length)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

        if porcupine.process(pcm) >= 0:
            print("Wake word detected!")
            break

    audio_stream.close()
    pa.terminate()


def transcribe_audio(filename):
    """
    Transcribe the specified audio file using OpenAI's Whisper.
    """
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    with open(filename, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
            language="en",
            prompt = "Transcribe the following audio clip in one or two words:"
        )

    return transcript


def record_until_pause(threshold=500, pause_duration=3):
    """
    Continuously record audio from the microphone until a pause is detected.
    
    :param threshold: The volume threshold below which is considered silence.
    :param pause_duration: The duration of silence in seconds to consider as a pause.
    """
    pa = pyaudio.PyAudio()

    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000,
                     input=True, frames_per_buffer=1024)

    print("Start speaking...")

    frames = []
    silent_frames = 0
    pause_frames = int(16000 / 1024 * pause_duration)
    
    while True:
        data = stream.read(1024)
        frames.append(data)

        # Check volume
        amplitude = np.frombuffer(data, np.int16)
        volume = np.sqrt(np.mean(amplitude**2))

        if volume < threshold:
            silent_frames += 1
        else:
            silent_frames = 0

        if silent_frames >= pause_frames:
            print("Pause detected, processing audio.")
            break

    stream.stop_stream()
    stream.close()
    pa.terminate()

    filename = "output.wav"
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
    wf.close()

    return filename


def main():
    try:
        # First, we listen for a wake word
        detect_wake_word()
        print("Wake word detected. Now recording until a pause is detected...")

        # Once the wake word is detected, we start recording until a pause is detected
        recorded_audio_filename = record_until_pause()
        print(f"Recorded audio saved as {recorded_audio_filename}. Now transcribing...")

        # Finally, we transcribe the recorded audio
        transcription = transcribe_audio(recorded_audio_filename)
        print("Transcription:", transcription)

    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()
