import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Load environment variables from .env
load_dotenv()

# 2. Instantiate the Gemini chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
    )

# 3. Call it with a single message and print the result
response = llm.invoke("Hello, how can I prepare for a job interview?")

print(response.content)
