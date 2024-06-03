import requests

class Multimodal:
    def __init__(self):
        self.url_model_selection = "http://127.0.0.1:8000/model_selection"
        self.url_gpt = "http://127.0.0.1:8001/gpt"
        self.url_diffusion = "http://127.0.0.1:8002/diffusion"
    
    def model_selection(self):
        response = requests.get(self.url_model_selection)
        return response.json()
    
    def gpt(self, prompt: str):
        response = requests.post(self.url_gpt + f"?prompt={prompt}")
        return response.json()
    
    def diffusion(self, prompt: str):
        response = requests.post(self.url_diffusion, json={"prompt": prompt})
        return response.json()
    
    def multimodal(self, prompt: str):
        model = self.model_selection()["model"]
        print(f"Selected model: {model}")
        if model == "GPT":
            return self.gpt(prompt)
        elif model == "Diffusion": # Example of adding another model
            return self.diffusion(prompt)
        else:
            return {"error": "Model not found!"}
        
    def run(self):
        while True:
            prompt = input("Enter the prompt: ")
            response = self.multimodal(prompt)
            print(response['response'])

if __name__ == "__main__":
    Multimodal().run()