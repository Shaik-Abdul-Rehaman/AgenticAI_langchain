from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Prompt setup
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

parser = StrOutputParser()
chain = prompt_template | model | parser

# Define input schema
class TranslationInput(BaseModel):
    language: str
    text: str

# FastAPI app
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server using Langchain runnable interfaces"
)



from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello, World"}

add_routes(
    app,
    chain.with_types(input_type=TranslationInput),
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
