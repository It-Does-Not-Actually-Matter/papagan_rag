import os
import torch
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
##
load_dotenv()
PDF_FOLDER = "data"
DB_DIR = "./chroma_db"
os.makedirs(PDF_FOLDER, exist_ok=True)

def initialize_vectorstore():
    """Initializing Vector Database."""
    print("Initializing Vector Database...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    if os.path.exists(DB_DIR):
        print("Loading existing ChromaDB...")
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    else:
        print("Creating new ChromaDB from PDFs...")
        files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))[:5] 
        documents = []
        for f in files:
            try:
                documents.extend(PyPDFLoader(f).load())
            except Exception as e:
                print(f"Skipping {f}: {e}")
        
        if documents:
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
            docs = splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=DB_DIR)
        else:
            print("No PDF documents found.")
            vectorstore = None

    return vectorstore

def create_rag_chain(vectorstore):
    if not vectorstore: return None
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = Ollama(model="llama3:8b", temperature=0.1)
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a helpful AI assistant.
        Answer the user's question using ONLY the provided CONTEXT.
        Do not make assumptions or use outside knowledge.
        If the answer is not found in the context, say "Bu konuda bağlamda bilgi bulunamadı."
        Respond nicely and strictly in Turkish.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        ANSWER (in Turkish):
        """
    )
    
    return (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
         "question": RunnablePassthrough()}
        | prompt 
        | llm 
        | StrOutputParser()
    )

def main():
    vectorstore = initialize_vectorstore()
    rag_chain = create_rag_chain(vectorstore)
    
    if not rag_chain:
        print("Failed to initialize RAG chain. Please check your data.")
        return

    print("\n")
    print("PAPAGAN")
    print("\n")

    while True:
        try:
            user_input = input("\nUser: ")
            if not user_input.strip(): continue
            if user_input.lower() in ["exit", "quit"]: break
            
            print("Papagan:", end="", flush=True)
            for chunk in rag_chain.stream(user_input):
                print(chunk, end="", flush=True)
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()