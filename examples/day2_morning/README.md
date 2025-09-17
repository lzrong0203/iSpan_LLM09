# RAG System with Gemma3/Llama3 and BGE Embeddings

## 📦 Installation

### Quick Install (Minimal)
```bash
pip install -r requirements-minimal.txt
```

### Full Install (All features)
```bash
pip install -r requirements.txt
```

### Manual Install
```bash
# Essential packages
pip install sentence-transformers faiss-cpu pypdf numpy requests

# For local Llama3 (optional)
pip install torch transformers accelerate
```

## 🚀 Usage

### Using Gemma3 with Ollama (Recommended)
```bash
# 1. Ensure Ollama is running
ollama serve

# 2. Run the RAG system
python gemma3_rag_ollama.py
```

### Using Llama3 (requires model access)
```bash
python llama3_rag_system.py
```

## 📁 Files

- `gemma3_rag_ollama.py` - RAG with Gemma3 4B via Ollama
- `llama3_rag_system.py` - RAG with Llama3 (local or Ollama)
- `requirements.txt` - Full dependencies
- `requirements-minimal.txt` - Minimal dependencies
- `data/*.pdf` - Source PDF documents

## 🔧 Configuration

The system uses:
- **Embedding Model**: BAAI/bge-large-en-v1.5 (1024 dimensions)
- **LLM**: Gemma3 4B or Llama3
- **Vector DB**: FAISS
- **Chunk Size**: 500 words with 50 word overlap

## 📊 Data Sources

The system processes PDF files from the `data/` directory:
- 2305.14325v1.pdf
- 2502.14767v2.pdf
- 2506.08292v1.pdf
- 2509.05396v1.pdf

## 💡 Features

- PDF text extraction and chunking
- Semantic embedding generation
- Vector similarity search
- Context-aware answer generation
- Source attribution
- Interactive Q&A interface