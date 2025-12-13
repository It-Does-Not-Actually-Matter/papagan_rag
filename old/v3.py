import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import sounddevice as sd
import librosa
import threading
from collections import Counter
from queue import Queue
from dataclasses import dataclass
from typing import Optional, Tuple
import time

print("--- Mikrofon Listesi ---")
print(sd.query_devices())
sd.default.device = 5  


@dataclass
class VADConfig:
    def get_statistics(self):
        """İstatistikleri yazdır"""
        print("\n" + "="*60)
        print("📊 İSTATİSTİKLER")
        print("="*60)
        print(f"Toplam transkripsiyon: {self.stats['total_transcriptions']}")
        print(f"Başarılı: {self.stats['successful']}")
        print(f"Hallucination: {self.stats['hallucinations']}")
        if self.stats['total_transcriptions'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_transcriptions']) * 100
            print(f"Başarı oranı: {success_rate:.1f}%")
            print(f"Ortalama süre: {self.stats['avg_duration']:.2f}s")
        print("="*60)
    
    def save_history(self, filename: str = "transcription_history.txt"):
        """Transkripsiyon geçmişini dosyaya kaydet"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("TRANSKRIPSIYON GEÇMIŞI\n")
                f.write("="*60 + "\n\n")
                for i, trans in enumerate(self.transcription_history, 1):
                    f.write(f"[{i}] {trans.text}\n")
                    f.write(f"    Süre: {trans.duration:.2f}s | Hallucination: {trans.is_hallucination}\n\n")
            print(f"✅ Geçmiş kaydedildi: {filename}")
        except Exception as e:
            print(f"❌ Kayıt hatası: {e}")
    """Voice Activity Detection ayarları"""
    silence_threshold: float = 0.01
    silence_duration: float = 1.5  # saniye
    min_speech_duration: float = 0.3  # minimum konuşma süresi
    block_duration: float = 0.5


@dataclass
class TranscriptionResult:
    """Transkripsiyon sonucu"""
    text: str
    duration: float
    is_hallucination: bool
    hallucination_reason: Optional[str] = None


class WhisperTranscriber():
    def __init__(self, model_id: str = "openai/whisper-large-v3-turbo", vad_config: Optional[VADConfig] = None):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vad_config = vad_config or VADConfig()
        self.processing_queue = Queue()
        self.is_processing = False
        self.transcription_history = []
        self.stats = {
            "total_transcriptions": 0,
            "successful": 0,
            "hallucinations": 0,
            "avg_duration": 0
        }
        print(f"Using Device: {self.device}\nUsing Model: {self.model_id}")

    def load_model(self):
        """Model ve pipeline'ı yükle"""
        print("🔄 Model yükleniyor...")
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device=self.device
        )

        print("✅ Model ve Pipeline hazır!")
    
    def _is_hallucination(self, text: str) -> tuple[bool, str]:
        """
        Hallucination (tekrar/saçmalama) kontrolü
        Returns: (is_hallucination, reason)
        """
        words = text.split()
        
        # Boş veya çok kısa metinler
        if len(words) < 3:
            return False, None
        
        # 1. Unique kelime oranı kontrolü
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # %30'dan az unique kelime
                return True, f"Tekrar oranı çok düşük: {unique_ratio:.2f}"
        
        # 2. Ardışık kelime çiftleri kontrolü
        if len(words) > 8:
            pairs = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            pair_counts = Counter(pairs)
            most_common_pair, count = pair_counts.most_common(1)[0]
            if count > 5:  # Aynı 2 kelime grubu 5+ kez
                return True, f"'{most_common_pair}' {count}x tekrar"
        
        # 3. Aynı kelimenin art arda tekrarı (3+ kez)
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2]:
                return True, f"'{words[i]}' kelimesi art arda tekrar"
        
        return False, None
    
    def _process_audio(self, audio_data: np.ndarray, sample_rate: int):
        """
        Ses verisini işle ve transkribe et (ayrı thread'de çalışır)
        """
        if self.is_processing:
            print("\n⚠️  Bir transkripsiyon devam ediyor, bekleyin...")
            return
        
        self.is_processing = True
        start_time = time.time()
        
        try:
            # Minimum süre kontrolü
            audio_duration = len(audio_data) / 16000  # 16kHz'de
            if audio_duration < self.vad_config.min_speech_duration:
                print(f"\n⚠️  Ses çok kısa ({audio_duration:.1f}s), atlanıyor")
                return
            
            # Whisper 16kHz ister, gerekirse resample yap
            if sample_rate != 16000:
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sample_rate, 
                    target_sr=16000
                )

            # Whisper ile transkripsiyon
            result = self.pipe(
                audio_data,
                generate_kwargs={
                    "language": "turkish",
                    "task": "transcribe",
                    "no_repeat_ngram_size": 3,
                    "temperature": 0.0,
                    "compression_ratio_threshold": 2.4,  # Hallucination kontrolü
                    "logprob_threshold": -1.0,
                    "no_speech_threshold": 0.6,
                }
            )
            
            text = result["text"].strip()
            duration = time.time() - start_time
            
            # Hallucination kontrolü
            is_hallu, reason = self._is_hallucination(text)
            
            # İstatistikler güncelle
            self.stats["total_transcriptions"] += 1
            if is_hallu:
                self.stats["hallucinations"] += 1
            else:
                self.stats["successful"] += 1
            
            # Ortalama süre güncelle
            total = self.stats["total_transcriptions"]
            self.stats["avg_duration"] = (
                (self.stats["avg_duration"] * (total - 1) + duration) / total
            )
            
            # Transkripsiyon sonucu oluştur
            transcription = TranscriptionResult(
                text=text,
                duration=duration,
                is_hallucination=is_hallu,
                hallucination_reason=reason
            )
            
            # Geçmişe ekle
            self.transcription_history.append(transcription)
            
            # Çıktı
            if is_hallu:
                print(f"\n⚠️  Hallucination: {reason}")
            elif len(text) > 0:
                print(f"\n🗣️  Metin: {text}")
                print(f"⏱️  Süre: {duration:.2f}s")
            else:
                print("\n⚠️  Metin algılanamadı")
                
        except Exception as e:
            print(f"\n❌ İşleme hatası: {e}")
        finally:
            self.is_processing = False
    
    def transcribe(self, audio_path: str):
        """Dosyadan transkripsiyon yap"""
        if not hasattr(self, "pipe") or self.pipe is None:
            self.load_model()

        print(f"📁 Dosya işleniyor: {audio_path}")

        transcription = self.pipe(
            audio_path,
            chunk_length_s=30,
            batch_size=8,
            return_timestamps=True
        )
    
        print("✅ Transkripsiyon tamamlandı!")
        return transcription
    
    def mic_transcribe(self):
        """Mikrofondan gerçek zamanlı transkripsiyon (VAD ile)"""
        # Model kontrolü
        if not hasattr(self, "pipe") or self.pipe is None:
            self.load_model()
        
        print("\n" + "="*60)
        print("🎤 GERÇEK ZAMANLI TRANSKRİPSİYON")
        print("="*60)
        print("💡 Konuşun, susun -> Otomatik işlenecek")
        print("🛑 Durdurmak için: Ctrl+C")
        print("="*60 + "\n")
        
        # Ayarlar
        DEVICE_ID = 5
        device_info = sd.query_devices(DEVICE_ID, 'input')
        sample_rate = int(device_info['default_samplerate'])
        INPUT_CHANNELS = device_info['max_input_channels']
        block_duration = 0.5
        
        print(f"🔧 Mikrofon: {device_info['name']}")
        print(f"🔧 Ayarlar: {sample_rate}Hz, {INPUT_CHANNELS} kanal")
        print("="*60 + "\n")

        # VAD parametreleri
        silence_threshold = self.vad_config.silence_threshold
        silence_limit = int(self.vad_config.silence_duration / block_duration)
        
        # Adaptif eşik için
        noise_floor = 0  # Gürültü tabanı
        noise_samples = []
        
        # State değişkenleri
        sentence_buffer = []
        silence_counter = 0
        is_speaking = False
        
        # Debug
        volume_history = []
        max_volume = 0

        try:
            # İlk 40 blokta (20 saniye) gürültü tabanını öğren
            print("🔊 Gürültü seviyesi ölçülüyor, lütfen sessiz kalın...")
            for i in range(40):
                recording = sd.rec(
                    int(block_duration * sample_rate),
                    samplerate=sample_rate,
                    channels=INPUT_CHANNELS,
                    dtype='float32',
                    device=DEVICE_ID
                )
                sd.wait()
                audio_chunk = np.mean(recording, axis=1) if INPUT_CHANNELS > 1 else recording.flatten()
                volume = np.sqrt(np.mean(audio_chunk**2))
                noise_samples.append(volume)
                print(f"\r{'█' * (i+1)}", end="", flush=True)
            
            # Gürültü tabanını hesapla (ortalamanın 2 katı)
            noise_floor = np.mean(noise_samples) * 2
            silence_threshold = max(noise_floor, 0.0005)  # Minimum 0.002
            
            print(f"\n✅ Gürültü tabanı: {noise_floor:.4f}")
            print(f"✅ Ses eşiği ayarlandı: {silence_threshold:.4f}")
            print("\n🎤 Artık konuşabilirsiniz!\n")
            
            while True:
                # Ses kaydı al
                recording = sd.rec(
                    int(block_duration * sample_rate),
                    samplerate=sample_rate,
                    channels=INPUT_CHANNELS,
                    dtype='float32',
                    device=DEVICE_ID
                )
                sd.wait()
                
                # Stereo -> Mono
                audio_chunk = np.mean(recording, axis=1) if INPUT_CHANNELS > 1 else recording.flatten()
                
                # Ses seviyesi ölç
                volume = np.sqrt(np.mean(audio_chunk**2))
                
                # Debug: Ses takibi
                volume_history.append(volume)
                if volume > max_volume:
                    max_volume = volume
                
                # Periyodik ses raporu
                if len(volume_history) % 20 == 0:
                    avg_vol = np.mean(volume_history[-20:])
                    print(f"📊 Ses seviyesi - Şu an: {volume:.4f} | Ortalama: {avg_vol:.4f} | Maksimum: {max_volume:.4f}")

                # VAD: Ses var mı?
                if volume > silence_threshold:
                    if not is_speaking:
                        print("\n🎤 Konuşma başladı...", end=" ", flush=True)
                        is_speaking = True
                        sentence_buffer = []
                    
                    silence_counter = 0
                    sentence_buffer.append(audio_chunk)
                    print("🔊", end="", flush=True)
                
                # Sessizlik var
                else:
                    if is_speaking:
                        silence_counter += 1
                        sentence_buffer.append(audio_chunk)

                        # Cümle bitti mi?
                        if silence_counter >= silence_limit:
                            print("\n⏳ Cümle tamamlandı, işleniyor...")
                            
                            # Buffer'ı birleştir
                            full_sentence = np.concatenate(sentence_buffer)
                            
                            # Ayrı thread'de işle (bloklamadan devam et)
                            processing_thread = threading.Thread(
                                target=self._process_audio,
                                args=(full_sentence.copy(), sample_rate),
                                daemon=True
                            )
                            processing_thread.start()
                            
                            # State'i sıfırla (hemen dinlemeye devam et)
                            sentence_buffer = []
                            is_speaking = False
                            silence_counter = 0
                            print("👂 Tekrar dinleniyor...\n")

        except KeyboardInterrupt:
            print("\n\n" + "="*60)
            print("🛑 Program sonlandırılıyor...")
            print("="*60)
            
            # İstatistikleri göster
            self.get_statistics()
            
            # Geçmişi kaydet
            if len(self.transcription_history) > 0:
                save = input("\n💾 Transkripsiyon geçmişini kaydetmek ister misiniz? (e/h): ")
                if save.lower() == 'e':
                    self.save_history()
                    
        except Exception as e:
            print(f"\n❌ HATA: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Özel VAD ayarlarıyla başlat (opsiyonel)
    vad_config = VADConfig(
        silence_threshold=0.01,
        silence_duration=1.5,
        min_speech_duration=0.3
    )
    
    transcriber = WhisperTranscriber(vad_config=vad_config)
    transcriber.mic_transcribe()