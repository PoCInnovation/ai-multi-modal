from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from example_model import Transformer
class GPT:
    def __init__(self):
        self.model = Transformer(
            src_vocab_size=32,
            trg_vocab_size=32,
            src_pad_idx=0,
            trg_pad_idx=0,
            embed_size=512,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            max_length=100,
        )
        # Remplacez par votre chemin d'accès complet au fichier .pth sinon cela ne marchera pas...
        self.model.load_state_dict(torch.load("/home/ramosleandre/delivery/poc/ai-multi-modal/Models/inversion_model.pth"))
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval()
        self.unique_words = []

    def extract_unique_words(self, prompt: str):
        all_words = set()
        for word in prompt.split():
            all_words.add(word)
        sorted_unique_words = sorted(all_words)
        self.unique_words = sorted_unique_words
    
    def tokenize(self, prompt: str):
        verse = {word: i for i, word in enumerate(self.unique_words)}
        return [verse[word] if word in verse else self.model.src_pad_idx for word in prompt.split()]

    def detokenize(self, indices):
        inverse = {i: word for i, word in enumerate(self.unique_words)}
        return ' '.join([inverse[index] for index in indices if index in inverse])

    def generate_response(self, prompt: str):
        self.extract_unique_words(prompt)

        src = self.tokenize(prompt)
        trg = [self.model.trg_pad_idx]
        src = torch.tensor(src).unsqueeze(0).to(self.model.device)
        trg = torch.tensor(trg).unsqueeze(0).to(self.model.device)
    
        with torch.no_grad():
            for _ in range(25):
                trg_mask = self.model.make_trg_mask(trg)
                out = self.model(src, trg)
                pred_token = out.argmax(2)[:, -1].item()
                trg = torch.cat([trg, torch.tensor([[pred_token]], device=self.model.device)], dim=1)
                if pred_token == self.model.trg_pad_idx:
                    break
        return self.detokenize(trg.squeeze(0).tolist())

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
