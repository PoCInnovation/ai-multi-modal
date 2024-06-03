from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VerificationRequest(BaseModel):
    result: str

def verify_result(result: str) -> int:
    if "valid" in result:
        return 1
    else:
        return 0

@app.get("/")
async def root():
    return {"message": "Model verification API is running!"}

@app.post("/model_verification")
async def model_verification(request: VerificationRequest):
    verification_result = verify_result(request.result)
    return {"verification_result": verification_result}
