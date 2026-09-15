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
        "You are a helpful user service agent "
        "whose job is to help users with their inquiries, "
        "only regarding the products and services offered by our company."
    ),
    ("user", "{user_input}")
])

# 4. Start a loop to continuously accept user input and generate responses

chat_history = []
print("Welcome to the Customer Service Chatbot! Type your query below (type 'exit' to quit):") 

while True:
    user_input = input("your query: ")
    if user_input.lower() == "exit":
        print("Exiting the chatbot. Goodbye!")
        break

    chat_history.append({"role": "user", "content": user_input})
    # 4. Format the prompt with an actual input to see what it produces
    formatted = prompt.invoke({"user_input": chat_history[-1]["content"]})

    # 5. Call it with the formatted prompt and print the result
    response = llm.invoke(formatted)

    # 6. Print the response from the AI
    print("AI:" + response.content)
