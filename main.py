import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

def build_chain():
    """Constructs and returns the LCEL chain (llm, prompt, parser)."""

    # 1. Instantiate the Gemini chat model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    # 2. Build a ChatPromptTemplate with two roles:
    prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            "You are a helpful user service agent "
            "whose job is to help users with their inquiries, "
            "only regarding the products and services offered by our company."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{user_input}")
    ])

    # 3. Create a chain that combines the prompt, the LLM, and an output parser
    chain = prompt | llm | StrOutputParser()
    return chain

def get_response(chain, user_input: str, chat_history: list) -> str:
    """
    Takes the chain, current input, and existing history.
    Invokes the chain and returns just the AI's reply as a string.
    Does NOT mutate chat_history itself -- caller decides what to do with it.
    """
    response = chain.invoke({"user_input": user_input, "chat_history": chat_history})
    return response

if __name__ == "__main__":

    chain = build_chain()

    chat_history = []

    print("Welcome to the Customer Service Chatbot! Type your query below (type 'exit' to quit):") 
    while True:
        user_input = input("your query: ")
        if user_input.lower() == "exit":
            print("Exiting the chatbot. Goodbye!")
            break

        response = get_response(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        print("AI:" + response)
