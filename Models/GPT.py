from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

client = OpenAI()
import os

class GPT:
    def __init__(self):
        self.model = "gpt-3.5-turbo-0125"

    def generate_response(self, prompt: str):
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=100,
        temperature=0.7)
        return response.choices[0].message.content

app = FastAPI()

origins = ["*"]

Model = GPT()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "GPT API is running!"}

@app.post("/gpt")
async def gpt(prompt: str):
    return {"response": Model.generate_response(prompt)}
