import gradio as gr
import os
import glob
import torch
import whisper
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

# Klasörleri oluştur
pdf_folder = "data"
audio_folder = "ses_data"
os.makedirs(pdf_folder, exist_ok=True)
os.makedirs(audio_folder, exist_ok=True)

print("🚀 Sistem başlatılıyor...")

# Whisper modelini yükle
print("📥 Whisper modeli yükleniyor...")
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("medium", device=device)
print(f"✅ Whisper yüklendi ({device})")

# Embeddings modelini yükle
print("📥 Embedding modeli yükleniyor...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("✅ Embedding modeli yüklendi")

# Vector store'u yükle veya oluştur
def initialize_vectorstore():
    """Vector store'u başlat"""
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    
    if not pdf_files:
        print("⚠️ Henüz PDF dosyası yüklenmemiş!")
        return None
    
    print(f"📚 {len(pdf_files)} PDF dosyası bulundu")
    
    if os.path.exists("./chroma_db"):
        print("📂 Mevcut veritabanı yükleniyor...")
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
    else:
        print("🔨 Yeni veritabanı oluşturuluyor...")
        documents = []
        for pdf_path in pdf_files[:5]:  # İlk 5 PDF
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120
        )
        docs = text_splitter.split_documents(documents)
        
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
    
    print("✅ Veritabanı hazır")
    return vectorstore

vectorstore = initialize_vectorstore()

# LLM'i başlat
print("🤖 Ollama LLM başlatılıyor...")
llm = Ollama(
    model="llama3:8b",
    temperature=0.1
)






print("✅ LLM hazır")

# RAG chain'i oluştur
retriever = None

def create_rag_chain():
    """RAG chain'i oluştur"""
    global retriever
    
    if vectorstore is None:
        return None
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )
    
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
    
    return rag_chain

rag_chain = create_rag_chain()

print("✅ Sistem hazır!\n")


# Gradio fonksiyonları
def get_sources(question):
    """Soruyla ilgili kaynak dökümanları al"""
    if retriever is None:
        return []
    
    try:
        docs = retriever.get_relevant_documents(question)
        sources = []
        
        for i, doc in enumerate(docs):
            source_info = {
                "index": i + 1,
                "file": os.path.basename(doc.metadata.get("source", "Bilinmeyen")),
                "page": doc.metadata.get("page", "?"),
                "content": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            }
            sources.append(source_info)
        
        return sources
    except:
        return []


def format_sources_html(sources):
    """Kaynakları HTML olarak formatla"""
    if not sources:
        return ""
    
    html = '<div style="margin-top: 15px; padding: 15px; background: #f8f9fa; border-radius: 10px; border-left: 4px solid #667eea;">'
    html += '<div style="font-size: 14px; font-weight: 600; color: #374151; margin-bottom: 10px;">📚 Kaynaklar:</div>'
    html += '<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px;">'
    
    # Kaynak chip'leri
    for source in sources:
        html += f'''
        <span style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 6px 14px;
            border-radius: 16px;
            font-size: 13px;
            font-weight: 500;
            display: inline-block;
            box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3);
        ">
            📄 {source["file"]} - s.{source["page"]}
        </span>
        '''
    
    html += '</div>'
    
    # Kaynak detayları
    html += '<details style="margin-top: 8px;">'
    html += '<summary style="cursor: pointer; font-size: 13px; color: #6b7280; font-weight: 500;">🔍 Detayları göster</summary>'
    html += '<div style="margin-top: 10px;">'
    
    for source in sources:
        html += f'''
        <div style="
            margin-top: 10px;
            padding: 12px;
            background: white;
            border-radius: 6px;
            border: 1px solid #e5e7eb;
        ">
            <div style="font-size: 12px; font-weight: 600; color: #667eea; margin-bottom: 6px;">
                #{source["index"]} - {source["file"]} (Sayfa {source["page"]})
            </div>
            <div style="font-size: 12px; color: #4b5563; line-height: 1.5;">
                {source["content"]}
            </div>
        </div>
        '''
    
    html += '</div></details></div>'
    return html


def chat_response(message, history):
    """Chat mesajını işle ve cevap döndür"""
    if not message or not message.strip():
        return "⚠️ Lütfen bir soru yazın!"
    
    if rag_chain is None:
        return "⚠️ Lütfen önce PDF dosyaları yükleyin! 'Belge Yönetimi' sekmesinden PDF ekleyebilirsiniz."
    
    try:
        # Cevabı al
        response = rag_chain.invoke(message.strip())
        
        # Kaynakları al ve formatla
        sources = get_sources(message.strip())
        sources_html = format_sources_html(sources)
        
        # Cevap + kaynakları birleştir
        full_response = response + "\n\n" + sources_html
        
        return full_response
    except Exception as e:
        return f"❌ Hata oluştu: {str(e)}"


def transcribe_audio(audio):
    """Ses kaydını metne çevir"""
    if audio is None:
        return None
    
    try:
        # Ses dosyasını kaydet
        audio_path = os.path.join(audio_folder, "temp_recording.wav")
        
        # Gradio audio'dan gelen tuple: (sample_rate, audio_data)
        import soundfile as sf
        if isinstance(audio, tuple):
            sample_rate, audio_data = audio
            sf.write(audio_path, audio_data, sample_rate)
        else:
            audio_path = audio
        
        # Whisper ile transkripsiyonu al
        result = whisper_model.transcribe(audio_path, language="tr", fp16=False)
        question = result['text'].strip()
        
        return question
    
    except Exception as e:
        return f"❌ Transkripsiyon hatası: {str(e)}"


def upload_pdf(files):
    """PDF dosyalarını yükle ve veritabanını güncelle"""
    global vectorstore, rag_chain
    
    if not files:
        return "⚠️ Lütfen PDF dosyası seçin!"
    
    try:
        uploaded_count = 0
        
        for file in files:
            # Dosyayı data klasörüne kopyala
            import shutil
            file_name = os.path.basename(file.name)
            dest_path = os.path.join(pdf_folder, file_name)
            shutil.copy(file.name, dest_path)
            uploaded_count += 1
        
        # Veritabanını yeniden oluştur
        vectorstore = initialize_vectorstore()
        rag_chain = create_rag_chain()
        
        return f"✅ {uploaded_count} PDF dosyası başarıyla yüklendi ve veritabanı güncellendi!"
    
    except Exception as e:
        return f"❌ Hata: {str(e)}"


def get_pdf_list():
    """Yüklü PDF dosyalarının listesini al"""
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    if not pdf_files:
        return "📚 Henüz PDF yüklenmedi"
    
    file_list = "\n".join([f"• {os.path.basename(f)}" for f in pdf_files])
    return f"📚 Yüklü PDF Dosyaları ({len(pdf_files)}):\n\n{file_list}"


# Gradio arayüzü
with gr.Blocks(
    title="🦜 Papağan RAG",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        /* Header stil */
        .app-header {
            text-align: center;
            padding: 30px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
        .app-header h1 {
            font-size: 42px;
            margin: 0 0 8px 0;
            font-weight: 700;
        }
        .app-header p {
            font-size: 16px;
            margin: 0;
            opacity: 0.95;
        }
        /* Chat mesajlarını büyüt */
        .message {
            font-size: 16px !important;
            line-height: 1.6 !important;
        }
        /* Input alanını büyüt */
        .input-area textarea {
            font-size: 16px !important;
            min-height: 60px !important;
        }
        /* Butonları büyüt */
        button {
            font-size: 15px !important;
            font-weight: 500 !important;
        }
        /* Tab'ları büyüt */
        .tab-nav button {
            font-size: 16px !important;
            padding: 10px 16px !important;
        }
        /* Chatbot alanını genişlet */
        .chatbot {
            height: 600px !important;
        }
    """
) as app:
    
    # Header
    gr.HTML("""
        <div class="app-header">
            <h1>🦜 Papağan RAG</h1>
            <p>Yapay Zeka Destekli Belge Asistanı - ChatGPT Tarzı Arayüz</p>
        </div>
    """)
    
    with gr.Tabs():
        # Chat Sekmesi
        with gr.Tab("💬 Sohbet"):
            gr.Markdown("""
            ### 💡 Nasıl Kullanılır:
            - 💬 Kaysın bir chat gibi soru sorun, cevapları message bubble'larda görün
            - 🎤 Ses kaydı yapıp metne çevirebilirsiniz
            - 📚 Her cevabın altında hangi kaynaklardan bilgi alındığını görebilirsiniz
            """)
            
            # Chat Interface
            chatbot = gr.Chatbot(
                label="Sohbet Geçmişi",
                height=600,
                show_copy_button=True
            )
            
            with gr.Row():
                with gr.Column(scale=4):
                    msg = gr.Textbox(
                        label="Mesajınız",
                        placeholder="Sorunuzu buraya yazın... (Enter ile gönder)",
                        lines=2,
                        max_lines=5,
                        show_label=False,
                        container=False
                    )
                with gr.Column(scale=1):
                    audio_record = gr.Audio(
                        label="🎤 Ses Kaydı",
                        type="numpy",
                        sources=["microphone"],
                        show_label=False
                    )
            
            with gr.Row():
                transcribe_btn = gr.Button("🎙️ Sesi Metne Çevir", variant="secondary", size="sm")
                clear = gr.Button("�️ Sohbeti Temizle", variant="stop", size="sm")
            
            # Chat fonksiyonları
            def respond(message, chat_history):
                bot_message = chat_response(message, chat_history)
                chat_history.append((message, bot_message))
                return "", chat_history
            
            # Mesaj gönderme
            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            
            # Ses → Metin
            transcribe_btn.click(
                fn=transcribe_audio,
                inputs=audio_record,
                outputs=msg
            )
            
            # Temizleme
            clear.click(lambda: None, None, chatbot, queue=False)
            
            gr.Markdown("""
            ---
            **� İpuçları:**
            - Uzun sohbetlerde "Sohbeti Temizle" ile yeni başlayabilirsiniz
            - Ses kaydından sonra metni düzenleyebilirsiniz
            - Her cevabın altındaki kaynaklara tıklayarak detayları görebilirsiniz
            """)
        
        # PDF Yönetimi Sekmesi
        with gr.Tab("📁 Belge Yönetimi"):
            gr.Markdown("## 📚 PDF Belgelerini Yönetin")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📤 Yeni PDF Yükle")
                    pdf_upload = gr.File(
                        label="PDF Dosyalarını Seçin",
                        file_types=[".pdf"],
                        file_count="multiple",
                        height=150
                    )
                    upload_button = gr.Button(
                        "📤 Yükle ve İşle",
                        variant="primary",
                        size="lg"
                    )
                    upload_status = gr.Textbox(
                        label="Durum",
                        lines=3,
                        interactive=False
                    )
                
                with gr.Column():
                    gr.Markdown("### 📋 Sistemdeki PDF'ler")
                    pdf_list_button = gr.Button(
                        "🔄 Listeyi Yenile",
                        size="lg"
                    )
                    pdf_list = gr.Textbox(
                        label="Yüklü Dosyalar",
                        lines=12,
                        interactive=False,
                        value=get_pdf_list()
                    )
            
            upload_button.click(
                fn=upload_pdf,
                inputs=pdf_upload,
                outputs=upload_status
            ).then(
                fn=get_pdf_list,
                outputs=pdf_list
            )
            
            pdf_list_button.click(
                fn=get_pdf_list,
                outputs=pdf_list
            )
            
            gr.Markdown("""
            ---
            ### ℹ️ Bilgilendirme:
            - 📚 Birden fazla PDF dosyası yükleyebilirsiniz
            - 🔄 Yüklenen dosyalar otomatik olarak vektör veritabanına eklenir
            - ⚡ Maksimum 5 PDF dosyası işlenir (performans için)
            - 💾 Dosyalar `data` klasörüne kaydedilir
            - 🗄️ Vektör veritabanı `chroma_db` klasöründe saklanır
            """)
        
        # Sistem Bilgisi Sekmesi
        with gr.Tab("ℹ️ Bilgi"):
            gr.Markdown("""
            # 🦜 Papağan RAG Hakkında
            
            ## 🔧 Sistem Bileşenleri
            
            ### 🎤 Ses Tanıma
            - **Model:** OpenAI Whisper (Medium)
            - **Dil:** Türkçe optimizasyonlu
            - **Cihaz:** """ + device + """
            
            ### 🤖 Dil Modeli
            - **Model:** Llama 3 (8B parametreli)
            - **Sıcaklık:** 0.1 (tutarlı cevaplar için)
            - **Çalıştırma:** Ollama üzerinden
            
            ### 📊 Embedding ve Arama
            - **Embedding Modeli:** BAAI/bge-m3 (Çok dilli)
            - **Vektör Veritabanı:** ChromaDB
            - **Chunk Boyutu:** 800 karakter
            - **Overlap:** 120 karakter
            - **Arama:** En benzer 6 doküman
            
            ### 🎨 Arayüz
            - **Framework:** Gradio
            - **Tema:** Soft (Blue/Indigo)
            - **Font:** Inter (Google Font)
            
            ## � Kullanım Akışı
            
            1. **PDF Yükleme:** Belgelerinizi sisteme ekleyin
            2. **Vektörleştirme:** Belgeler otomatik olarak parçalanır ve vektörlere dönüştürülür
            3. **Soru Sorma:** Metin veya ses ile soru sorun
            4. **RAG Süreci:**
               - Sorunuz vektöre dönüştürülür
               - En ilgili belge parçaları bulunur
               - Bu parçalar LLM'e context olarak verilir
               - LLM sadece bu context'i kullanarak cevap üretir
            5. **Kaynak Gösterimi:** Hangi belgelerden bilgi alındığı gösterilir
            
            ## ⚡ Özellikler
            
            - ✅ ChatGPT tarzı chat arayüzü
            - ✅ Sesli soru sorma
            - ✅ Ses → metin dönüştürme
            - ✅ Kaynak gösterimi (citation chips)
            - ✅ Sohbet geçmişi
            - ✅ Çoklu PDF desteği
            - ✅ Türkçe optimizasyon
            - ✅ ASCII-only output (uyumluluk için)
            
            ## 🎯 En İyi Sonuçlar İçin
            
            - Spesifik ve net sorular sorun
            - PDF'lerinizin metin formatında olmasına dikkat edin (taranmış görüntüler değil)
            - Sistemde ilgili belgeler olduğundan emin olun
            - Uzun sohbetlerde bazen temizleme yapmak performansı artırır
            
            ---
            
            **Geliştirici Notu:** Bu sistem tamamen lokal çalışır. Verileriniz dışarı çıkmaz.
            """)
    
    # Footer
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #6b7280; font-size: 14px;">
        <p>🦜 Papağan RAG v1.0 | Powered by LangChain, Whisper & Llama 3</p>
    </div>
    """)


# Uygulamayı başlat
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
