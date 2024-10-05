from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

client = OpenAI()

class ModelSelection:
    def __init__(self, models_list, configuration_prompt):
        self.models_list = models_list
        self.configuration_prompt = configuration_prompt

    def model_selection(self, prompt: str):
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": self.configuration_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=5,
        temperature=0.7)
        return response.choices[0].message.content

models_list = ["GPT", "Diffusion", "text_completion"]

configuration_prompt = """
You are a model selection assistant. You will be given a prompt and you will have to choose the correct model based on the prompt.

Criteria for selection:
- GPT: Use GPT for tasks involving text generation, conversation, or any language-based tasks. Examples include writing poems, stories, or assisting with text-based queries.
- Diffusion: Use Diffusion for tasks involving image generation or manipulation. Examples include generating photos, paintings, or any visual content.
- text_completion: Use text_completion ONLY when the prompt explicitly asks to complete a sentence or text. Examples include "Complete the following sentence: ..." or "Finish the phrase ...".

Examples:
1. Prompt: "Hello, can you help me with a poem?"
   GPT
2. Prompt: "Generate a photo of a cat"
   Diffusion
3. Prompt: "Can you tell me a story about a brave knight?"
   GPT
4. Prompt: "Create an illustration of a fantasy landscape"
   Diffusion
5. Prompt: "Complete the following sentence: 'Roses are red, violets are blue...'"
   text_completion
6. Prompt: "Can you explain what a large language model (LLM) is?"
   GPT

You can only answer using one of the following words: 'text_completion', 'Diffusion', 'GPT'.

Evaluate the following prompt and choose the correct model:

Prompt: "{prompt_here}"
"""


Model = ModelSelection(models_list, configuration_prompt)

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
    return {"message": "Model Selection API is running!"}

@app.post("/model_selection")
async def model_selection(input_data: dict):
    input_data["model"] = Model.model_selection(input_data["prompt"])
    return input_data