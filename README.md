# RAG Sistemi

Bu proje, bilgisayarinizdaki PDF dosyalarini okuyup, icine sorular sorabileceginiz yerel bir yapay zeka sistemidir. Sistem sadece PDF icerigini kullanir, disaridan bilgi eklemez.

## Sistem Mimarisi

![RAG Architecture](https://github.com/It-Does-Not-Actually-Matter/papagan_rag/raw/main/rag.jpg)

Sistem 5 temel asamadan olusur:
1. **Document Loading**: PDF dosyalari yuklenir
2. **Splitting**: Belgeler kucuk parcalara bolunur
3. **Storage**: Parcalar vektorstore'da saklanir
4. **Retrieval**: Sorguya en uygun parcalar bulunur
5. **Output**: LLM ile cevap uretilir

## Ne Yapar?

- `data/` klasorundeki ilk 5 PDF dosyasini otomatik yukler
- Metinleri 800 karakterlik parcalara ayirir (120 karakter bindirme ile)
- BAAI/bge-m3 modeli ile embeddings olusturur
- Chroma vektorstore kullanarak benzerligi hesaplar
- Sorularinizi PDF icerigine gore cevaplar
- PDF'te yoksa su cevabi verir: `Baglamda cevap bulunamadi.`

## Gereksinimler

### Ollama Kurulu Olmali

Bu proje LLM olarak Ollama kullanir. Calistirmadan once:

```bash
ollama pull llama3:8b
```

Ollama arka planda calisir durumda olmali.

### Python Kutuphaneleri

Sanal ortam olusturun ve aktif edin:

```bash
python -m venv env
source env/bin/activate  # Linux/Mac
# veya
env\Scripts\activate  # Windows
```

Kutuphaneleri yukleyin:

```bash
pip install -r requirements.txt
```

## Kurulum

1. Projeyi klonlayin:
```bash
git clone https://github.com/It-Does-Not-Actually-Matter/papagan_rag.git
cd papagan_rag
```

2. `data/` klasoru olusturun ve PDF dosyalarinizi icine atin:
```bash
mkdir -p data
# PDF dosyalarinizi data/ klasorune kopyalayin
```

3. Sanal ortam olusturun ve bagimlilikari yukleyin:
```bash
python -m venv env
source env/bin/activate  # Linux/Mac veya env\Scripts\activate (Windows)
pip install -r requirements.txt
```

4. Ollama'nin calistigini dogrulayin:
```bash
ollama list
```

## Kullanim

### Terminal Modu

Projeyi calistirin:

```bash
python main.py
```

Terminal acildiginda sorular sorabilirsiniz:

```
---RAG---
Kullanici: Yazilim tasarimi nedir?
Cevap: ...

Kullanici: Bu belgede ne anlatiliyor?
Cevap: ...

Kullanici: exit
```

Cikmak icin `exit` yazin.

### API Modu (Open WebUI Entegrasyonu)

RAG sistemini OpenAI-uyumlu bir API olarak calistirabilirsiniz:

```bash
python api.py
```

API sunucusu `http://localhost:8000` adresinde calisir.

#### API Endpoint'leri

| Endpoint | Method | Aciklama |
|----------|--------|----------|
| `/v1/models` | GET | Mevcut modelleri listeler |
| `/v1/chat/completions` | POST | Sohbet tamamlama (RAG ile) |
| `/health` | GET | Saglik kontrolu |

#### Open WebUI ile Kullanim

1. Open WebUI'yi Docker ile kurun:

```powershell
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 -e OPENAI_API_KEY=dummy -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
```

2. Tarayicinizda `http://localhost:3000` adresine gidin
3. Hesap olusturun (yerel, sadece sizin icin)
4. Settings > Connections > OpenAI API bolumunde:
   - API Base URL: `http://host.docker.internal:8000/v1`
   - API Key: `dummy`
5. Model olarak `papagan-rag` secin ve kullanmaya baslayin

## Teknik Detaylar

- **Embedding Model**: BAAI/bge-m3
- **LLM**: Llama3 8B (Ollama)
- **Vector Store**: Chroma
- **Chunk Size**: 800 karakter
- **Chunk Overlap**: 120 karakter
- **Retrieval**: Top-6 similarity search
- **Temperature**: 0.3 (daha deterministik cevaplar)

## Ozellikler

- Yerel calisir, internet gerektirmez
- Sadece verdiginiz PDF'lerden cevap verir
- Turkce cevaplar uretir (ASCII karakterler ile)
- Chroma DB persist ile tekrar yukleme gerektirmez
- Hallusinasyon onleme mekanizmasi
- OpenAI-uyumlu API ile Open WebUI entegrasyonu
- FastAPI tabanli REST API

## Dosya Yapisi

```
papagan_rag/
├── main.py              # Terminal modu (CLI)
├── api.py               # API modu (Open WebUI entegrasyonu)
├── requirements.txt     # Python bagimliliklar
├── README.md           # Dokumantasyon
├── data/               # PDF dosyalari (sizin olusturacaginiz)
├── chroma_db/          # Vektorstore (otomatik olusur)
└── .env                # Ortam degiskenleri (opsiyonel)
```

## Lisans

MIT License