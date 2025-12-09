from langchain_community.document_loaders import PyPDFLoader , PyPDFDirectoryLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import glob , os

load_dotenv()

pdf_folder = "data"
pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
pdf_files = pdf_files[:5] 

documents = []
for pdf_path in pdf_files:
    loader = PyPDFLoader(pdf_path)
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120
)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

if os.path.exists("./chroma_db"):
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
else:
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

llm = Ollama(
    model="llama3:8b",
    temperature=0.3
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

# Dökümanları formatlama fonksiyonu
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template = """
You are a retrieval-based assistant.
Answer the user's question using ONLY the given CONTEXT.
Do NOT use external knowledge.
Do NOT make assumptions or hallucinate.

Rules:
- Write the answer in TURKISH using ONLY ASCII characters.
- Do NOT use Turkish characters like: ç, ğ, ş, ı, İ, ö, ü.
- The answer should be CLEAR, EXPLANATORY, and 3 to 6 sentences long.
- You may rephrase the context but do NOT add new information.
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
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("---RAG---")
while True:
    query = input("Kullanici: ")
    if query.lower() == "exit":
        break

    try:
        response = rag_chain.invoke(query)
        print(f"\nCevap: {response}\n")
    except Exception as e:
        print(f"Query error: {e}")