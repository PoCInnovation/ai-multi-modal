import requests

class Multimodal:
    def __init__(self):
        self.url_model_selection = "http://127.0.0.1:8000/model_selection"
        self.url_gpt = "http://127.0.0.1:8001/gpt"
        self.url_diffusion = "http://127.0.0.1:8002/diffusion"
        self.url_model_verification = "http://127.0.0.1:8003/model_verification"
    
    def model_selection(self):
        response = requests.get(self.url_model_selection)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None
    
    def gpt(self, prompt: str):
        response = requests.post(self.url_gpt + f"?prompt={prompt}")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None
    
    def diffusion(self, prompt: str):
        response = requests.post(self.url_diffusion, json={"prompt": prompt})
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None

    def model_verification(self, response: str):
        result = requests.get(self.url_model_verification, json={"result": response})
        return result.json()

    def multimodal(self, prompt: str):
        model_selection_result = self.model_selection()
        if not model_selection_result:
            return {"error": "Model selection failed"}

        model = model_selection_result["model"]
        print(f"Selected model: {model}")

        if model == "GPT":
            response = self.gpt(prompt)
        elif model == "Diffusion": # Example of adding another model
            response = self.diffusion(prompt)
        else:
            return {"error": "Model not found"}
        verification_result = self.model_verification(response['response'])
        if not verification_result:
            return {"error": "Model verification failed"}
        
        return verification_result
            
    def run(self):
        while True:
            prompt = input("Enter the prompt: ")
            response = self.multimodal(prompt)

            print(response['response'])

if __name__ == "__main__":
    Multimodal().run()