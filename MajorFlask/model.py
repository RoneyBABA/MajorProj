# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

#Step1: Setup GROQ API key
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")

#Step2: Convert image to required format
import base64

if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY is not set! Add it to your environment or .env file.")

#image_path="D:/College/Major/ai-doctor-2.0-voice-and-vision/skin_rash.jpg"
def encode_image(image_path):   
    image_file=open(image_path, "rb")
    return base64.b64encode(image_file.read()).decode('utf-8')

#Step3: Setup Multimodal LLM 
from groq import Groq

model="meta-llama/llama-4-maverick-17b-128e-instruct"

def analyze_image_with_query(query, model, encoded_image):
    client=Groq(api_key=GROQ_API_KEY)  
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ],
        }]
    chat_completion=client.chat.completions.create(
        messages=messages,
        model=model
    )

    return (chat_completion.choices[0].message.content)

def analyze_query(query, model):
    client=Groq(api_key=GROQ_API_KEY)  
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": query
                },
            ],
        }]
    chat_completion=client.chat.completions.create(
        messages=messages,
        model=model
    )

    return (chat_completion.choices[0].message.content)