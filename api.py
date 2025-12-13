from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid
import time

# main.py'den RAG bileşenlerini import edin
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
import glob, os

app = FastAPI(title="Papagan RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG bileşenlerini başlat
pdf_folder = "data"
pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))[:5]

documents = []
for pdf_path in pdf_files:
    loader = PyPDFLoader(pdf_path)
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name=" ",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

if os.path.exists("./chroma_db"):
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
else:
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="./chroma_db")

llm = Ollama(model="llama3:8b", temperature=0.3)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a retrieval-based assistant.
Answer the user's question using ONLY the given CONTEXT.
Do NOT use external knowledge.

Rules:
- Write the answer in TURKISH using ONLY ASCII characters.
- The answer should be CLEAR, EXPLANATORY, and 3 to 6 sentences long.
- If the answer is not found in the context, respond exactly with:
  "Baglamda cevap bulunamadi."

CONTEXT:
{context}

QUESTION:
{question}

DETAILED ASCII TURKISH ANSWER:
"""
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# OpenAI-uyumlu API modelleri
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "papagan-rag"
    messages: List[Message]
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "papagan-rag", "object": "model", "owned_by": "local"}]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        # Son kullanıcı mesajını al
        user_message = next(
            (msg.content for msg in reversed(request.messages) if msg.role == "user"),
            None
        )
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # RAG chain'i çalıştır
        response = rag_chain.invoke(user_message)
        
        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model="papagan-rag",
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": response},
                "finish_reason": "stop"
            }]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
