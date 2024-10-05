import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive environments

import requests
from config import api_map
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
        print(input_data)
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
        image_path = "temp.png"  # Use a fixed path for testing
        with open(image_path, "wb") as f:
            f.write(image_data)
        return image_path

    def display_answer(self, response: dict):
        if response["Type"] == "Text":
            print(response["Response"])
        elif response["Type"] == "Image":
            image_path = response["Response"]
            try:
                img = plt.imread(image_path)
                plt.imshow(img)
                plt.axis("off")
                plt.savefig("output_image.png", bbox_inches='tight', pad_inches=0)  # Save the figure instead of showing it
                print("Image saved as 'output_image.png'")
            except Exception as e:
                print(f"Failed to display image: {e}")
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
