from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import json

client = OpenAI()

class Moderation:
    def __init__(self, prompt):
        self.prompt = prompt

    def moderation(self):
        response = client.moderations.create(input=self.prompt)
        output = response.results[0]
        return output

    def serialize(self, output):
        """Recursively walk object's hierarchy."""
        if isinstance(output, (bool, int, float, str)):
            return output
        elif isinstance(output, dict):
            output = output.copy()
            for key in output:
                output[key] = self.serialize(output[key])
            return output
        elif isinstance(output, list):
            return [self.serialize(item) for item in output]
        elif isinstance(output, tuple):
            return tuple(self.serialize(item) for item in output)
        elif hasattr(output, '__dict__'):
            return self.serialize(output.__dict__)
        else:
            return repr(output)  # Don't know how to handle, convert to string


prompt = input("Enter the prompt: ")

modo = Moderation(prompt)

output = modo.moderation()

serialized_output = modo.serialize(output)

print(serialized_output)
