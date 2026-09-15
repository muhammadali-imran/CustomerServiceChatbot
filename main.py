import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Load environment variables from .env
load_dotenv()

# 2. Instantiate the Gemini chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
    )

# 3. Build a ChatPromptTemplate with two roles:
prompt = ChatPromptTemplate.from_messages([
    (
        "system", 
        "You are a helpful customer service agent "
        "whose job is to help customers with their inquiries, "
        "only regarding the products and services offered by our company."
    ),
    ("human", "{user_input}")
])

# 4. Format the prompt with an actual input to see what it produces
user_input = input("Enter your query to the customer service agent: ")
formatted = prompt.invoke({"user_input": user_input})

# 5. Call it with the formatted prompt and print the result
response = llm.invoke(formatted)

print(response.content)
