# Papagan RAG

A local RAG chatbot that answers questions from your PDF documents. Runs entirely offline with Ollama.

## Features

- Text and voice input (Whisper)
- Turkish language responses
- Incremental PDF indexing
- Parallel document loading

## Quick Start

```bash
# Install Ollama and pull the model
ollama pull llama3:8b

# Setup Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add PDFs and run
mkdir data
cp your-files.pdf data/
python main.py
```

## Usage

```
Type text or 'v' for voice (exit to quit): What is machine learning?
Papagan: [Answers based on your PDFs...]
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Llama3 8B (Ollama) |
| Embeddings | BAAI/bge-m3 |
| Vector DB | ChromaDB |
| Voice | OpenAI Whisper |

## License

MIT