from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import base64
import os
import requests
import openai

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

class Diffusion:
    def __init__(self):
        self.image_path = "generated_image.png"  # Local path for the generated image

    def generate_response(self, prompt: str):
        return self.create_image(prompt)

    def create_image(self, prompt: str):
        try:
            # Call the OpenAI API to generate an image
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size="256x256",
                response_format="url"  # Use "url" to get the URL of the generated image
            )
            image_url = response['data'][0]['url']

            # Download the image
            self.download_image(image_url)

            return self.image_path  # Return the local path of the downloaded image
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def download_image(self, image_url: str):
        # Download the image from the URL
        img_data = requests.get(image_url).content
        with open(self.image_path, 'wb') as handler:
            handler.write(img_data)

    def image_base64(self, image_path: str):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

app = FastAPI()

origins = ["*"]
Model = Diffusion()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Diffusion API is running!"}

@app.post("/diffusion")
async def diffusion(input_data: dict):
    prompt = input_data.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    generated_image = Model.generate_response(prompt)

    try:
        base64_image = Model.image_base64(generated_image)
        return {"Type": "Image", "Response": generated_image, "Base64": base64_image}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
