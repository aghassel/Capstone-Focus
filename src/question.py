from groq import Groq
from wake_words import detect_wake_word, transcribe_audio, record_until_pause
import asyncio
import dotenv

async def question_mode(transcript):
    dotenv.load_dotenv()
    groq = Groq(api_key='') # add key
    print(transcript)
    chat_completion = groq.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "you are a helpful assistant. provide brief responses in around 10 words."
            },
            {
                "role": "user",
                "content": f"{transcript}",
            }
        ],

        model="llama2-70b-4096",
        #model = "mixtral-8x7b-32768",

        max_tokens=100,
    )
    return chat_completion.choices[0].message.content
