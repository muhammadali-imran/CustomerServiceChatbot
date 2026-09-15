from fastapi import FastAPI
from pydantic import BaseModel
from main import build_chain, get_response
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI()
chain = build_chain()
sessions: dict[str, list] = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    session_id = request.session_id
    message = request.message

    if session_id not in sessions:
        sessions[session_id] = []

    chat_history = sessions[session_id]
    response = get_response(chain, message, chat_history)
    chat_history.append(HumanMessage(content=message))
    chat_history.append(AIMessage(content=response))
    return ChatResponse(reply= response)
    

