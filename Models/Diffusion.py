from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import base64


# TODO: Replace with a reel implementation
class Diffusion:
    def __init__(self):
        self.image_path = "image.png"

    def generate_response(self, prompt: str):
        return self.image_path
    
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
    prompt = input_data["prompt"]
    generated_image = Model.generate_response(prompt)

    try:
        base64_image = Model.image_base64(generated_image)
        return {"Type": "Image", "Response": generated_image, "Base64": base64_image}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


