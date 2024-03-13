from groq import Groq
from wake_words import detect_wake_word, transcribe_audio, record_until_pause
import asyncio
import dotenv

async def question_mode(transcript):
    dotenv.load_dotenv()
    groq = Groq(api_key='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')
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
