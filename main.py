from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import glob
import os
import whisper
import torch
import pyaudio
import wave
import threading
import time
from pynput import keyboard

load_dotenv()

pdf_folder = "data"
audio_folder = "ses_data"
os.makedirs(audio_folder, exist_ok=True)

print("Loading Whisper model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("medium", device=device)
print(f"Whisper loaded on {device}\n")

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
    temperature=0.1
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
    template="""
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


def process_query(query):
    """Your existing query processing logic"""
    try:
        response = rag_chain.invoke(query)
        return response
    except Exception as e:
        return f"Query error: {e}"


def run_cli():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    p = pyaudio.PyAudio() 

    recording_active = False
    frames = []
    stream = None

    def on_press(key):
        nonlocal recording_active, frames, stream

        try:
            if key == keyboard.Key.space:
                if not recording_active:
                    recording_active = True
                    frames = []
                    print("\n🎤 Speak...")

                    stream = p.open(
                        format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK
                    )

                    def record_thread():
                        while recording_active:
                            data = stream.read(CHUNK)
                            frames.append(data)

                    thread = threading.Thread(target=record_thread, daemon=True)
                    thread.start()

                else:
                    recording_active = False
                    print("  Recording stopped, processing...")

                    time.sleep(0.2)

                    if stream:
                        stream.stop_stream()
                        stream.close()

                    if len(frames) > 0:
                        for file in glob.glob(os.path.join(audio_folder, "*.wav")):
                            try:
                                os.remove(file)
                            except:
                                pass

                        audio_file = os.path.join(audio_folder, "recording.wav")

                        try:
                            wf = wave.open(audio_file, 'wb')
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(p.get_sample_size(FORMAT))
                            wf.setframerate(RATE)
                            wf.writeframes(b''.join(frames))
                            wf.close()

                            print("  Transcribing with Whisper...")
                            result = whisper_model.transcribe(audio_file, language="tr", fp16=False)
                            question = result['text'].strip()

                            if question:
                                print(f"\nKullanici: {question}")
                                print("🤔 RAG system thinking...")
                                response = process_query(question)
                                print(f"\nCevap: {response}\n")
                            else:
                                print("  No audio detected!\n")

                        except Exception as e:
                            print(f"  Error: {e}\n")
                    else:
                        print("  Could not save audio!\n")

            elif key == keyboard.Key.esc:
                print("\nExiting...")
                p.terminate()
                return False

        except AttributeError:
            pass

    print("=" * 60)
    print("---RAG with Whisper---")
    print("=" * 60)
    print("\nModes:")
    print("  - SPACE = Start/Stop voice recording")
    print("  - Type 'text' to switch to text mode")
    print("  - ESC = Exit\n")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        while True:
            try:
                query = input("Kullanici: ")
                if query.lower() == "exit":
                    break
                if query.lower() == "text":
                    print("Text mode active. Type 'voice' to return to voice mode.\n")
                    continue

                response = process_query(query)
                print(f"\nCevap: {response}\n")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    finally:
        p.terminate()
        listener.stop()
        print("\nGoodbye!")


if __name__ == "__main__":
    run_cli()