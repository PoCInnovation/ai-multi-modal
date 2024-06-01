from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

models_list = ["GPT", "Diffusion"]

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

# TODO: Need to be replaced with a POST request to select the model based on the input
@app.get("/model_selection")
async def model_selection():
    return {"model": models_list[0]}