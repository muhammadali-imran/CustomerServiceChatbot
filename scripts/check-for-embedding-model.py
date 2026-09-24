import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("Available embedding models for your API key:")
for model in client.models.list():
    # Check if the model supports embedding content
    if hasattr(model, 'supported_actions') and 'embedContent' in model.supported_actions:
        print(f" - {model.name}")

