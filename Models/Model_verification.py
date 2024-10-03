from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

client = OpenAI()

class ModelVerification:
    def __init__(self, configuration_prompt):
        self.configuration_prompt = configuration_prompt

    def model_verification(self, response: str):
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": self.configuration_prompt},
            {"role": "user", "content": response},
        ],
        max_tokens=5,
        temperature=0.7)
        return response.choices[0].message.content
    
configuration_prompt = """
You are a model verification assistant. You will be given the response of a model, and you need to verify if the response is valid or invalid based on whether it could hurt the user's sensitivity. 

Criteria for validation:
- Valid: The response is respectful, neutral, or positive and does not contain offensive language.
- Invalid: The response contains offensive language, disrespectful comments, or anything that could hurt the user's sensitivity.

Examples:
1. Response: "Hello! I'm just a computer program, so I don't have feelings, but I'm here and ready to help you. How can I assist you today?"
   valid
2. Response: "Fuck you!"
   invalid

You can only answer using one of the following words: 'valid', 'invalid'.

Evaluate the following response according to these criteria:

Response: "{response_here}"
"""

Model = ModelVerification(configuration_prompt)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Model verification API is running!"}

@app.post("/model_verification")
async def model_verification(response: dict):
    if response["Type"] == "Text":
        response["verified"] = True
        text_verification = Model.model_verification(response["Response"])
        if text_verification == 'valid':
            response["verified"] = True
        else:
            response["verified"] = False
    else:
        response["verified"] = True
    return response

