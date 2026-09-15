import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagePlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.schema import HumanMessage, AIMessage

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
    MessagePlaceholder(variable_name="chat_history"),
    ("user", "{user_input}")
])

chain = prompt | llm | StrOutputParser()

# 4. Start a loop to continuously accept user input and generate responses

chat_history = []
print("Welcome to the Customer Service Chatbot! Type your query below (type 'exit' to quit):") 

while True:
    user_input = input("your query: ")
    if user_input.lower() == "exit":
        print("Exiting the chatbot. Goodbye!")
        break

    # 5. Invoke the chain with the user input to see what it produces
    response = chain.invoke({"user_input": user_input, "chat_history": chat_history})
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))

    # 6. Print the response from the AI
    print("AI:" + response)
