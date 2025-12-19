import os
import torch
import glob
import warnings
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
import tempfile
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.layout import Layout
import sys

# LangChain deprecation uyarılarını bastır
warnings.filterwarnings("ignore", category=DeprecationWarning)
##
load_dotenv()

PDF_FOLDER = "data"
DB_DIR = "./chroma_db"
os.makedirs(PDF_FOLDER, exist_ok=True)

def validate_file_constraints(current_count, new_files):
    MAX_BATCH = 50
    MAX_TOTAL = 200
    
    # 1. Check: Batch Size
    if len(new_files) > MAX_BATCH:
        print(f"\n[ERROR] Tek seferde en fazla {MAX_BATCH} dosya yüklenebilir.")
        print(f"Tespit edilen yeni dosya: {len(new_files)}")
        excess = len(new_files) - MAX_BATCH
        print(f"Lütfen {excess} adet dosyayı siliniz.")
        print("Yeni dosyalar (ilk 10):")
        for f in new_files[:10]:
            print(f" - {os.path.basename(f)}")
        if len(new_files) > 10: print(" ...")
        return False

    # 2. Check: Total Size
    if current_count + len(new_files) > MAX_TOTAL:
        print(f"\n[ERROR] Maksimum {MAX_TOTAL} dosya limitine ulaşıldı.")
        print(f"Mevcut: {current_count}, Eklenecek: {len(new_files)}")
        print("Lütfen dosya sayısını azaltınız.")
        return False
        
    return True

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
        
        # Check for new files to add incrementally
        print("Checking for new files...")
        existing_data = vectorstore.get()
        existing_sources = set()
        if existing_data and 'metadatas' in existing_data:
            for m in existing_data['metadatas']:
                if m and 'source' in m:
                    existing_sources.add(m['source'])
        
        # Get all current files
        all_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
        new_files = [f for f in all_files if f not in existing_sources]
        
        if new_files:
            if not validate_file_constraints(len(existing_sources), new_files):
                print("Skipping new file addition due to constraints.")
                return vectorstore

            print(f"Found {len(new_files)} new files. Adding to DB...")
            new_docs_content = []
            for f in new_files:
                try:
                    new_docs_content.extend(PyPDFLoader(f).load())
                except Exception as e:
                    print(f"Skipping {f}: {e}")
            
            if new_docs_content:
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
                chunks = splitter.split_documents(new_docs_content)
                vectorstore.add_documents(chunks)
                print(f"Successfully added {len(new_files)} new files.")
        else:
            print("No new files to add.")
    else:
        print("Creating new ChromaDB from PDFs...")
        files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
        
        if not validate_file_constraints(0, files):
            return None

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
    llm = Ollama(model="llama3:8b", temperature=0.3)
    
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

WHISPER_MODEL = None

def get_voice_input():
    global WHISPER_MODEL
    try:
        if WHISPER_MODEL is None:
            print("Loading Whisper model...")
            WHISPER_MODEL = whisper.load_model("medium")

        fs = 44100
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
        
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        path = tmp_file.name
        tmp_file.close()
        
        sf.write(path, audio, fs)
        
        print("Transcribing...")
        audio_data, sr = sf.read(path)
        audio_data = audio_data.flatten().astype(np.float32)
        
        if sr != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        
        result = WHISPER_MODEL.transcribe(audio_data, language="tr")
        
        try:
            os.remove(path)
        except Exception:
            pass
        
        return result["text"]
    except Exception as e:
        print(f"Voice input error: {e}")
        import traceback
        traceback.print_exc()
        return ""


def display_welcome_screen():
    """Görsel hoş geldiniz ekranını göster."""
    console = Console()
    
    # Ana başlık
    papagan_title = """
    ██████╗  █████╗ ██████╗  █████╗  ██████╗  █████╗ ███╗   ██╗
    ██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝ ██╔══██╗████╗  ██║
    ██████╔╝███████║██████╔╝███████║██║  ███╗███████║██╔██╗ ██║
    ██╔═══╝ ██╔══██║██╔═══╝ ██╔══██║██║   ██║██╔══██║██║╚██╗██║
    ██║     ██║  ██║██║     ██║  ██║╚██████╔╝██║  ██║██║ ╚████║
    ╚═╝     ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝
    """
    
    title_text = Text(papagan_title, style="bold cyan")
    
    # Açıklama metni
    description = Text(
        "🦜 Akıllı RAG Sistemi - Sorularınıza cevap vermeye hazır!",
        justify="center",
        style="bold yellow"
    )
    
    # Bilgilendirme paneli
    info_content = Text()
    info_content.append("💡 İpuçları:\n", style="bold green")
    info_content.append("  • Sorunuzu yazın ve Enter tuşuna basın\n")
    info_content.append("  • Çıkmak için ", style="white")
    info_content.append("exit", style="bold red")
    info_content.append(" veya ", style="white")
    info_content.append("quit", style="bold red")
    info_content.append(" yazın\n", style="white")
    info_content.append("  • ", style="white")
    info_content.append("Ctrl+C", style="bold magenta")
    info_content.append(" tuşu ile de çıkabilirsiniz", style="white")
    
    # Paneller oluştur
    console.print(Align.center(title_text))
    console.print(Align.center(description))
    console.print()
    console.print(Panel(
        info_content,
        border_style="cyan",
        title="[bold]Yardım[/bold]",
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
            "[bold red]Hata![/bold red] RAG zinciri başlatılamadı.\nLütfen verilerinizi kontrol edin.",
            border_style="red",
            title="[bold red]Başlatma Hatası[/bold red]"
        )
        console.print(error_panel)
        return

    display_welcome_screen()

    while True:
        try:
            choice = console.input("[bold cyan]👤 Type text or 'v' for voice (q to quit):[/bold cyan] ").strip()
            
            if not choice:
                continue
            
            if choice.lower() == 'q':
                farewell = Panel(
                    "[bold yellow]Görüşmek üzere! 🦜[/bold yellow]",
                    border_style="yellow",
                    title="[bold]Hoşça Kalın[/bold]"
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
            
            # Papagan cevabı
            console.print("[bold magenta]🦜 Papagan:[/bold magenta] ", end="", soft_wrap=True)
            response_text = ""
            for chunk in rag_chain.stream(user_input):
                response_text += chunk
                console.print(chunk, end="", soft_wrap=True)
            console.print()
            console.print()
            
        except KeyboardInterrupt:
            console.print()
            farewell = Panel(
                "[bold yellow]Program sonlandırıldı. Hoşça kalın! 👋[/bold yellow]",
                border_style="yellow",
                title="[bold]Çıkış[/bold]"
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