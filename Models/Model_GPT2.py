##
## EPITECH PROJECT, 2024
## ai-multi-modal
## File description:
## Model_GPT2
##

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Model_GPT2:
    def __init__(self, model_name="gpt2", max_length=100, temperature=0.7, top_k=50, top_p=0.9):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


    def generate_response(self, input_text: str):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=100,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1
        )

        predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return predicted_text

Model = Model_GPT2()

@app.get("/")
async def root():
    return {"message": "GPT2 API is running!"}

@app.post("/text_completion")
async def gpt2(input_data: dict):
    # to call the model we need to have {"prompt": "text"}
    prompt = input_data["prompt"]
    return {"Type": "Text", "Response": Model.generate_response(prompt)}


