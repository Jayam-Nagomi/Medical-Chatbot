from fastapi import FastAPI
from pydantic import BaseModel
from chatbot.chatbot import chatbot_response

app = FastAPI()

class MessageRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: MessageRequest):
    response, tag = chatbot_response(request.message)
    return {"response": response, "tag": tag}
