import pyaudio
import numpy as np
import threading
import time
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch

# ==================== AYARLAR ====================
SILENCE_THRESHOLD = 500      # Sessizlik eşiği (düşük = hassas)
SILENCE_DURATION = 1.5       # Kaç saniye sessizlik sonrası işle
SAMPLE_RATE = 16000          # Whisper için 16kHz
CHUNK_SIZE = 1024            # Ses buffer boyutu

# ==================== WHISPER MODEL YÜKLEME ====================
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

# Model cache'den yüklenecek (otomatik)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    low_cpu_mem_usage=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

print(f"✅ Model yüklendi ({device} üzerinde)")
print("=" * 60)

# ==================== SES KAYIT DEĞİŞKENLERİ ====================
audio_data = []
is_recording = False
last_sound_time = time.time()
lock = threading.Lock()

# ==================== PYAUDIO BAŞLAT ====================
p = pyaudio.PyAudio()

# Mevcut mikrofon listesini göster
print("\n🎤 Mikrofon Cihazları:")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f"  [{i}] {info['name']}")
print("=" * 60)

# ==================== SES SEVİYESİ HESAPLAMA ====================
def calculate_volume(audio_chunk):
    """Ses seviyesini hesapla (RMS - Root Mean Square)"""
    audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
    rms = np.sqrt(np.mean(audio_array**2))
    return rms

# ==================== WHISPER İLE TRANSKRİPSİYON ====================
def transcribe_audio(audio_frames):
    """Kaydedilen sesi Whisper ile metne çevir"""
    if not audio_frames:
        return
    
    print("\n🎙️  Ses işleniyor...")
    
    # Ses verilerini birleştir
    audio_bytes = b''.join(audio_frames)
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    
    # Whisper ile transkripsiyon
    try:
        result = pipe(
            {"sampling_rate": SAMPLE_RATE, "raw": audio_array},
            generate_kwargs={
                "language": "turkish",  # Türkçe için
                "task": "transcribe"
            }
        )
        text = result["text"].strip()
        
        if text:
            print(f"📝 Metin: {text}")
            print("=" * 60)
        else:
            print("⚠️  Metin algılanamadı")
    except Exception as e:
        print(f"❌ Hata: {e}")

# ==================== SES CALLBACK FONKSİYONU ====================
def audio_callback(in_data, frame_count, time_info, status):
    """PyAudio callback - mikrofon verisi geldiğinde çalışır"""
    global audio_data, is_recording, last_sound_time
    
    volume = calculate_volume(in_data)
    
    with lock:
        # Ses var mı?
        if volume > SILENCE_THRESHOLD:
            if not is_recording:
                print("\n🎤 Konuşma başladı...")
                is_recording = True
                audio_data = []
            
            audio_data.append(in_data)
            last_sound_time = time.time()
            
            # Ses seviyesi göstergesi
            bar_length = int(volume / 100)
            print(f"\r🔊 Ses: {'█' * min(bar_length, 50)}", end="", flush=True)
        
        # Sessizlik var ve kayıt yapılıyorsa
        elif is_recording:
            silence_time = time.time() - last_sound_time
            
            if silence_time >= SILENCE_DURATION:
                print("\n\n⏸️  Sessizlik algılandı, işleniyor...")
                is_recording = False
                
                # Ses verilerini kopyala ve işle
                frames_to_process = audio_data.copy()
                audio_data = []
                
                # Ayrı thread'de işle (bloklamadan devam etsin)
                threading.Thread(
                    target=transcribe_audio, 
                    args=(frames_to_process,),
                    daemon=True
                ).start()
    
    return (in_data, pyaudio.paContinue)

# ==================== MİKROFON AKIŞI BAŞLAT ====================
print("🎙️  Mikrofon dinleniyor... (Çıkmak için Ctrl+C)")
print("💡 Konuşun ve susun, sistem otomatik olarak metne çevirecek")
print("=" * 60)

stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=CHUNK_SIZE,
    stream_callback=audio_callback
)

stream.start_stream()

# ==================== ÇALIŞMAYA DEVAM ET ====================
try:
    while stream.is_active():
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n\n👋 Program sonlandırılıyor...")

# ==================== TEMİZLİK ====================
stream.stop_stream()
stream.close()
p.terminate()
print("✅ Temizlik tamamlandı, güle güle!")