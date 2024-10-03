from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests



class GPT:
    def __init__(self):
        self.text_model_url = "http://127.0.0.1:8004/text_model"

    def generate_response(self, prompt: str):
        response = requests.post(self.text_model_url, json={"prompt": prompt})
        if response.status_code == 200:
            return response.json()["Response"]
        else:
            return "Error in text model API call"

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
async def gpt(input_data: dict):
    prompt = input_data["prompt"]
    return {"Type": "Text", "Response": Model.generate_response(prompt)}

