import requests
from config import *
import matplotlib.pyplot as plt
import base64

class Multimodal:
    def __init__(self):
        self.api_map = api_map
    
    def api_request(self, model_name: str, input_data: dict):
        response = requests.post(self.api_map[model_name], json=input_data)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None
        
    def model_selection(self, input_data: dict):
        input_data = self.api_request("ModelSelection", input_data)
        if not input_data:
            return {"error": "Model selection failed"}
        
        if input_data["model"] == "Not Found":
            return {"error": "The model was not found"}
        return input_data
    
    def model_request(self, input_data: dict):
        response = self.api_request(input_data["model"], input_data)
        if not response:
            return {"error": "Model failed"}
        
        if "Base64" in response:
            response["Response"] = self.base64_to_image(response)
        return response
    
    def model_verification(self, response: dict):
        response = self.api_request("ModelVerification", response)
        
        if response["verified"] == False:
            return {"error": "Model not verified"}
        return response
    
    
    def multimodal(self, input_data: dict):
        input_data = self.model_selection(input_data)
        if "error" in input_data:
            return input_data

        response = self.model_request(input_data)
        if "error" in response:
            return response
        
        response = self.model_verification(response)
        if "error" in response:
            return response
        return response
    
    def base64_to_image(self, input_data: dict):
        image_data = base64.b64decode(input_data["Base64"])
        with open("temp.png", "wb") as f:
            f.write(image_data)
        return "temp.png"
    
    def display_answer(self, response: dict):
        if response["Type"] == "Text":
            print(response["Response"])
        elif response["Type"] == "Image":
            plt.imshow(plt.imread(response["Response"]))
            plt.axis("off")
            plt.show()
        else:
            print("Invalid response")
    
    def run(self):
        while True:
            prompt = input("Enter the prompt: ")
            input_data = {"prompt": prompt}
            if prompt.lower() == "exit":
                break
            response = self.multimodal(input_data)
            if "error" in response:
                print(response["error"])
            else:
                self.display_answer(response)

if __name__ == "__main__":
    ai_multi_modal = Multimodal()
    ai_multi_modal.run()
