import os
import torch
import glob
import warnings
import traceback
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
import librosa
import tempfile
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
import sys

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

PDF_FOLDER = "data"
DB_DIR = "./chroma_db"
os.makedirs(PDF_FOLDER, exist_ok=True)

_EMBEDDINGS_CACHE = None
WHISPER_MODEL = None


def get_embeddings():
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        _EMBEDDINGS_CACHE = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _EMBEDDINGS_CACHE


def validate_file_constraints(current_count, new_files):
    MAX_BATCH = 50
    MAX_TOTAL = 200
    
    if len(new_files) > MAX_BATCH:
        print(f"\n[ERROR] Tek seferde en fazla {MAX_BATCH} dosya yuklenebilir.")
        print(f"Tespit edilen yeni dosya: {len(new_files)}")
        excess = len(new_files) - MAX_BATCH
        print(f"Lutfen {excess} adet dosyayi siliniz.")
        print("Yeni dosyalar (ilk 10):")
        for f in new_files[:10]:
            print(f" - {os.path.basename(f)}")
        if len(new_files) > 10:
            print(" ...")
        return False

    if current_count + len(new_files) > MAX_TOTAL:
        print(f"\n[ERROR] Maksimum {MAX_TOTAL} dosya limitine ulasildi.")
        print(f"Mevcut: {current_count}, Eklenecek: {len(new_files)}")
        print("Lutfen dosya sayisini azaltiniz.")
        return False
        
    return True


def load_single_pdf(filepath):
    try:
        return PyPDFLoader(filepath).load()
    except Exception as e:
        print(f"Skipping {filepath}: {e}")
        return []


def load_pdfs_parallel(files, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(load_single_pdf, files)
    return [doc for docs in results for doc in docs]


def initialize_vectorstore():
    print("Initializing Vector Database...")
    embeddings = get_embeddings()

    if os.path.exists(DB_DIR):
        print("Loading existing ChromaDB...")
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        
        print("Checking for new files...")
        existing_data = vectorstore.get()
        existing_sources = set()
        if existing_data and 'metadatas' in existing_data:
            for m in existing_data['metadatas']:
                if m and 'source' in m:
                    existing_sources.add(os.path.abspath(m['source']))
        
        all_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
        new_files = [f for f in all_files if os.path.abspath(f) not in existing_sources]
        
        if new_files:
            if not validate_file_constraints(len(existing_sources), new_files):
                print("Skipping new file addition due to constraints.")
                return vectorstore

            print(f"Found {len(new_files)} new files. Adding to DB...")
            new_docs_content = load_pdfs_parallel(new_files)
            
            if new_docs_content:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_documents(new_docs_content)
                vectorstore.add_documents(chunks)
                print(f"Successfully added {len(new_files)} new files.")
        else:
            print("No new files to add.")
    else:
        print("Creating new ChromaDB from PDFs...")
        files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
        
        if not files:
            print("No PDF documents found in data folder.")
            return None
        
        if not validate_file_constraints(0, files):
            return None

        documents = load_pdfs_parallel(files)
        
        if documents:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=DB_DIR)
        else:
            print("No PDF documents could be loaded.")
            vectorstore = None

    return vectorstore


def create_rag_chain(vectorstore):
    if not vectorstore:
        return None
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model="llama3:8b", temperature=0.3)
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a helpful AI assistant.
        Answer the user's question using ONLY the provided CONTEXT.
        Do not make assumptions or use outside knowledge.
        If the answer is not found in the context, say "Bu konuda baglamda bilgi bulunamadi."
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


def get_voice_input():
    global WHISPER_MODEL
    try:
        if WHISPER_MODEL is None:
            print("Loading Whisper model...")
            WHISPER_MODEL = whisper.load_model("small")

        fs = 16000
        print("Press Enter to start recording...")
        input()
        print("Recording... Press Enter to stop.")

        recording = []
        def callback(indata, frames, time, status):
            recording.append(indata.copy())

        with sd.InputStream(samplerate=fs, channels=1, callback=callback):
            input()

        if not recording:
            return ""

        audio = np.concatenate(recording, axis=0)
        audio_data = audio.flatten().astype(np.float32)
        
        print("Transcribing...")
        result = WHISPER_MODEL.transcribe(audio_data, language="tr")
        
        return result["text"]
    except Exception as e:
        print(f"Voice input error: {e}")
        traceback.print_exc()
        return ""


def display_welcome_screen(console):
    papagan_title = """
    ██████╗  █████╗ ██████╗  █████╗  ██████╗  █████╗ ███╗   ██╗
    ██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝ ██╔══██╗████╗  ██║
    ██████╔╝███████║██████╔╝███████║██║  ███╗███████║██╔██╗ ██║
    ██╔═══╝ ██╔══██║██╔═══╝ ██╔══██║██║   ██║██╔══██║██║╚██╗██║
    ██║     ██║  ██║██║     ██║  ██║╚██████╔╝██║  ██║██║ ╚████║
    ╚═╝     ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝
    """
    
    title_text = Text(papagan_title, style="bold cyan")
    
    description = Text(
        "Papagan Chatbot - Sorulariniza cevap vermeye hazir!",
        justify="center",
        style="bold yellow"
    )
    
    info_content = Text()
    info_content.append("Ipuclari:\n", style="bold green")
    info_content.append("  - Sorunuzu yazin ve Enter tusuna basin\n")
    info_content.append("  - Cikmak icin ", style="white")
    info_content.append("exit", style="bold red")
    info_content.append(" yazin", style="white")
    
    console.print(Align.center(title_text))
    console.print(Align.center(description))
    console.print()
    console.print(Panel(
        info_content,
        border_style="cyan",
        title="[bold]Yardim[/bold]",
        expand=False,
        width=60
    ))
    console.print()


def main():
    console = Console()
    vectorstore = initialize_vectorstore()
    rag_chain = create_rag_chain(vectorstore)
    
    if not rag_chain:
        error_panel = Panel(
            "[bold red]Hata![/bold red] RAG zinciri baslatilamadi.\nLutfen verilerinizi kontrol edin.",
            border_style="red",
            title="[bold red]Baslatma Hatasi[/bold red]"
        )
        console.print(error_panel)
        return

    display_welcome_screen(console)

    while True:
        try:
            choice = console.input("[bold cyan]Type text or 'v' for voice (exit to quit):[/bold cyan] ").strip()
            
            if not choice:
                continue
            
            if choice.lower() == 'exit':
                farewell = Panel(
                    "[bold yellow]Gorusmek uzere![/bold yellow]",
                    border_style="yellow",
                    title="[bold]Hosca Kalin[/bold]"
                )
                console.print(farewell)
                break
            
            if choice.lower() == 'v':
                user_input = get_voice_input()
                console.print(f"[bold green]Transcribed:[/bold green] {user_input}")
            else:
                user_input = choice
            
            if not user_input.strip():
                continue
            
            console.print("[bold magenta]Papagan:[/bold magenta] ", end="", soft_wrap=True)
            for chunk in rag_chain.stream(user_input):
                console.print(chunk, end="", soft_wrap=True)
            console.print()
            console.print()
            
        except KeyboardInterrupt:
            console.print()
            farewell = Panel(
                "[bold yellow]Program sonlandirildi. Hosca kalin![/bold yellow]",
                border_style="yellow",
                title="[bold]Cikis[/bold]"
            )
            console.print(farewell)
            break
        except Exception as e:
            error_msg = Panel(
                f"[bold red]{str(e)}[/bold red]",
                border_style="red",
                title="[bold red]Hata[/bold red]"
            )
            console.print(error_msg)


if __name__ == "__main__":
    main()