import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import sounddevice as sd
import librosa

print("--- Mikrofon Listesi ---")
print(sd.query_devices())
sd.default.device = 1  # MME sürücüsü


class WhisperTranscriber():
    def __init__(self,model_id : str ="openai/whisper-large-v3-turbo"):
        self.model_id=model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using Device:{self.device}\nUsing Model: {self.model_id}")

    def load_model(self):

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            dtype=torch.float16,
            low_cpu_mem_usage = True,
            use_safetensors=True).to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model = self.model,
            tokenizer = self.processor.tokenizer,
            feature_extractor = self.processor.feature_extractor,
            processor = self.processor,
            device = self.device)

        print("Model ve Pipeline hazır!")
    
    def transcribe(self, audio_path : str):
        if not hasattr(self,"pipe") or self.pipe is None:
            self.load_model()

        print(f"File:{audio_path} Transcribing... ")

        transcription = self.pipe(
            audio_path,
            chunk_length_s = 30,
            batch_size = 8,
            return_timestamps = True)
    
        print("Transcription Done!")
        return transcription
    
    def mic_transcribe(self):
        # 1. Güvenlik Kontrolü
        if not hasattr(self, "pipe") or self.pipe is None:
            self.load_model()
        
        print("\n🎤 PRO MOD: Sadece konuştuğunda çevirecek... (Kapatmak için Ctrl+C)")
        print("-" * 50)
        
        # 2. AYARLAR
        DEVICE_ID = 1   # Internal Microphone - MME (daha yüksek ses seviyesi)
        
        # Cihazın doğal ayarlarını sorgula (WASAPI için kritik!)
        device_info = sd.query_devices(DEVICE_ID, 'input')
        sample_rate = int(device_info['default_samplerate'])
        INPUT_CHANNELS = device_info['max_input_channels']
        
        block_duration = 0.5
        
        print(f"🔧 Cihaz Ayarları: {sample_rate}Hz, {INPUT_CHANNELS} kanal")
        print(f"   Cihaz: {device_info['name']}")

        # PRO EŞİKLER
        silence_threshold = 0.01  # Ses eşiği
        silence_limit = 2  # 1 saniye sessizlik = cümle bitti (önceki 4'ten düşürdük) 
        
        # 3. HAFIZA
        sentence_buffer = []      
        silence_counter = 0       
        is_speaking = False
        
        # DEBUG: Ses seviyesi takibi
        volume_history = []
        max_volume = 0       

        try:
            while True: 
                # A) Sesi Al (Stereo olarak)
                recording = sd.rec(
                    int(block_duration * sample_rate), 
                    samplerate=sample_rate, 
                    channels=INPUT_CHANNELS, # 2 Kanal
                    dtype='float32',
                    device=DEVICE_ID # ID 18
                )
                sd.wait()
                
                # B) Stereo'dan Mono'ya Çevir (Veriyi Düzleştir)
                # İki kanalın ortalamasını alarak tek kanala düşürüyoruz
                audio_chunk = np.mean(recording, axis=1)
                
                # C) Ses Şiddetini Ölç
                volume = np.sqrt(np.mean(audio_chunk**2))
                
                # DEBUG: Ses seviyesini takip et
                volume_history.append(volume)
                if volume > max_volume:
                    max_volume = volume
                
                # Her 20 blokta bir ses seviyesi raporu (10 saniyede bir)
                if len(volume_history) % 20 == 0:
                    avg_vol = np.mean(volume_history[-20:])
                    print(f"\n📊 Ses: şu an={volume:.4f}, ort={avg_vol:.4f}, maks={max_volume:.4f}, eşik={silence_threshold:.4f}", end=" ", flush=True)

                # 5. KARAR MEKANİZMASI
                if volume > silence_threshold:
                    is_speaking = True
                    silence_counter = 0         
                    sentence_buffer.append(audio_chunk) 
                    print("🔊", end="", flush=True)  # Ses yakalandı işareti 
                    
                else:
                    if is_speaking:
                        silence_counter += 1
                        sentence_buffer.append(audio_chunk) 

                        if silence_counter >= silence_limit:
                            print("\n⏳ Cümle tamamlandı, işleniyor...")
                            
                            full_sentence = np.concatenate(sentence_buffer)
                            
                            # Model Çağrısı
                            # Whisper 16kHz ister, elimizdeki 44100Hz veriyi dönüştürüyoruz
                            if sample_rate != 16000:
                                full_sentence = librosa.resample(full_sentence, orig_sr=sample_rate, target_sr=16000)

                            result = self.pipe(
                                full_sentence,
                                language="tr", 
                                task="transcribe"
                            )
                            text = result["text"].strip()
                            
                            # Hallucination kontrolü (tekrarlanan kelime veya cümlecikler)
                            is_hallucination = False
                            words = text.split()
                            
                            # 1. Çok fazla tekrar var mı? (unique kelime oranı düşükse)
                            if len(words) > 10:
                                unique_ratio = len(set(words)) / len(words)
                                if unique_ratio < 0.3:  # %30'dan az unique kelime
                                    is_hallucination = True
                                    print(f"⚠️  Çok fazla tekrar (oran: {unique_ratio:.2f}), atlıyorum...\n")
                            
                            # 2. Ardışık kelime çiftleri tekrar ediyor mu?
                            if not is_hallucination and len(words) > 8:
                                # "Yavrum var" gibi 2 kelimelik grupları kontrol et
                                pairs = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
                                from collections import Counter
                                pair_counts = Counter(pairs)
                                most_common_pair, count = pair_counts.most_common(1)[0]
                                if count > 5:  # Aynı 2 kelime grubu 5+ kez tekrarlandı
                                    is_hallucination = True
                                    print(f"⚠️  Tekrarlayan ifade tespit edildi: '{most_common_pair}' ({count}x), atlıyorum...\n")
                            
                            if not is_hallucination and len(text) > 0:
                                print(f"🗣️  PRO: {text}\n")
                            
                            # Sıfırla
                            sentence_buffer = []
                            is_speaking = False
                            silence_counter = 0
                    else:
                        pass

        except KeyboardInterrupt:
            print("\n🛑 Mikrofon kapatıldı.")
        except Exception as e:
            print(f"\n❌ HATA OLUŞTU: {e}")

"""
    def mic_transcribe(self):
        if not hasattr(self,"pipe") or self.pipe is None:
            self.load_model()

        print("Microphone Transcribing... You can speak when you are ready! Press q to quit.")

        sample_rate = 16000
        noise_threshold = 0.05
        voice_control = 0.5
        silence_counter = 3
        is_speaking = False
        seq_buffer = []

        try:
            while True:
                audio = sd.rec(
                int(voice_control*sample_rate),
                samplerate = sample_rate,
                channels = 1,
                dtype = "float32")
                
                sd.wait()
                audio_chunk = np.flatten(audio)

                volume = np.sqrt(np.mean(audio_chunk**2))

                if volume > noise_threshold:
                    is_speaking = True
                    silence_counter = 0
                    seq_buffer = []
                    seq_buffer.append(audio_chunk)
                    while is_speaking:




        except KeyboardInterrupt:
            print("\nTranscription Stopped!")

"""
    
if __name__== "__main__":
    transcriber = WhisperTranscriber()
    transcriber.mic_transcribe()







