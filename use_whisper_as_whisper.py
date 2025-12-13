import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.utils import is_torch_sdpa_available
import numpy as np
import sounddevice as sd
import keyboard
import time
from datetime import datetime

print("="*60)
print("🎤 WHISPER PUSH-TO-TALK TRANSKRIPTOR")
print("="*60)

# SDPA kontrolü
if is_torch_sdpa_available():
    print("✅ SDPA (Scaled Dot-Product Attention) aktif")
else:
    print("⚠️  SDPA mevcut değil, varsayılan attention kullanılacak")

print("\n--- Mikrofon Listesi ---")
print(sd.query_devices())
sd.default.device = 1  # MME sürücüsü


class PushToTalkTranscriber:
    def __init__(self, model_id: str = "openai/whisper-large-v3-turbo"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_recording = False
        self.audio_buffer = []
        
        print(f"\n🔧 Cihaz: {self.device}")
        print(f"🔧 Model: {self.model_id}")
    
    def load_model(self):
        """Model yükle - SDPA ile optimize edilmiş"""
        print("\n🔄 Model yükleniyor...")
        
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # SDPA otomatik aktif (PyTorch 2.1.1+)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa"  # SDPA'yı açıkça belirt
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        # Pipeline - Chunked long-form için optimize
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self.device,
            chunk_length_s=30,  # 30 saniyelik chunk'lar
            batch_size=8,       # Batch processing
        )
        
        print("✅ Model hazır!")
    
    def record_audio(self, sample_rate=16000):
        """Ses kaydet - blocking"""
        print("\n🔴 KAYIT BAŞLADI (SPACE = Durdur)")
        self.audio_buffer = []
        self.is_recording = True
        
        def audio_callback(indata, frames, time_info, status):
            if self.is_recording:
                # Mono'ya çevir
                audio = np.mean(indata, axis=1) if indata.ndim > 1 else indata.flatten()
                self.audio_buffer.append(audio.copy())
        
        # Mikrofon ayarları
        device_info = sd.query_devices(1, 'input')
        channels = device_info['max_input_channels']
        
        # Stream başlat
        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype='float32',
            callback=audio_callback,
            device=1
        )
        
        stream.start()
        
        # SPACE tuşunu bekle
        keyboard.wait('space')
        
        self.is_recording = False
        stream.stop()
        stream.close()
        
        print("⏸️  KAYIT DURDURULDU")
        
        if len(self.audio_buffer) > 0:
            # Buffer'ı birleştir
            audio_data = np.concatenate(self.audio_buffer)
            duration = len(audio_data) / sample_rate
            print(f"⏱️  Kayıt süresi: {duration:.1f} saniye")
            return audio_data
        else:
            print("⚠️  Kayıt boş!")
            return None
    
    def transcribe_audio(self, audio_data):
        """Kaydedilen sesi transkribe et"""
        if audio_data is None or len(audio_data) == 0:
            print("❌ Ses verisi yok")
            return None
        
        print("\n🔄 İşleniyor...")
        start_time = time.time()
        
        try:
            # Whisper'a gönder - chunked long-form algoritması
            result = self.pipe(
                audio_data,
                generate_kwargs={
                    "language": "turkish",
                    "task": "transcribe",
                    "temperature": 0.0,
                    "compression_ratio_threshold": 2.4,
                    "logprob_threshold": -1.0,
                    "no_speech_threshold": 0.6,
                }
            )
            
            elapsed = time.time() - start_time
            text = result["text"].strip()
            
            if text:
                print("\n" + "="*60)
                print(f"📝 Metin: {text}")
                print(f"⏱️  İşlem süresi: {elapsed:.2f}s")
                print("="*60)
                return text
            else:
                print("⚠️  Metin algılanamadı")
                return None
                
        except Exception as e:
            print(f"❌ Hata: {e}")
            return None
    
    def run(self):
        """Ana döngü - Push-to-talk"""
        if not hasattr(self, 'pipe'):
            self.load_model()
        
        print("\n" + "="*60)
        print("🎤 PUSH-TO-TALK MOD")
        print("="*60)
        print("📌 SPACE: Kayıt başlat/durdur")
        print("📌 ESC: Programı kapat")
        print("="*60)
        
        try:
            while True:
                print("\n⏳ SPACE tuşuna basın...")
                keyboard.wait('space')
                
                # ESC kontrolü
                if keyboard.is_pressed('esc'):
                    print("\n👋 Çıkılıyor...")
                    break
                
                # Kayıt başlat
                audio = self.record_audio()
                
                # Transkribe et
                if audio is not None:
                    self.transcribe_audio(audio)
                
                # Kısa bekleme (tuş bounce için)
                time.sleep(0.3)
                
        except KeyboardInterrupt:
            print("\n\n👋 Program sonlandırıldı")
        except Exception as e:
            print(f"\n❌ Hata: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # PyTorch versiyonunu kontrol et
    print(f"\n🔧 PyTorch versiyon: {torch.__version__}")
    
    transcriber = PushToTalkTranscriber()
    transcriber.run()                        