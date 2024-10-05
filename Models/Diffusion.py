from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import base64
import os
import requests
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Diffusion:
    def __init__(self):
        self.image_path = "generated_image.png"  # Local path for the generated image

    def generate_response(self, prompt: str):
        # Upgrade the prompt using GPT-3.5
        upgraded_prompt = self.upgrade_prompt_with_gpt(prompt)
        print(f"Upgraded prompt: {upgraded_prompt}")
        return self.create_image(upgraded_prompt)

    def upgrade_prompt_with_gpt(self, prompt: str):
        """
        Use GPT-3.5 to upgrade the initial prompt.
        """
        try:
            # Use the chat completions API
            gpt_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant that helps improve prompts for image generation."},
                    {"role": "user", "content": f"Improve the following image generation prompt to make it more descriptive and creative:\n\n'{prompt}'"}
                ],
                max_tokens=50,  # Limit the length of the upgraded prompt
                temperature=0.7,  # Adjust temperature for more creativity
            )
            return gpt_response.choices[0].message.content
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error upgrading prompt: {str(e)}")

    def create_image(self, prompt: str):
        try:
            # Call the OpenAI API to generate an image
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            image_url = response.data[0].url

            # Download the image
            self.download_image(image_url)
            self.image_path = "generated_image.png"

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
