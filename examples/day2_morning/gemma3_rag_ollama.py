#!/usr/bin/env python3
"""
RAG System using Gemma3 4B and BGE-large-en-v1.5 embeddings
This version uses Ollama for running Gemma3:4b-it-qat model
"""

import os
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import json
import requests

# PDF processing
from pypdf import PdfReader

# Embeddings
from sentence_transformers import SentenceTransformer

# Vector database
import faiss

# Text processing
import re
from dataclasses import dataclass, asdict


@dataclass
class Document:
    """Document chunk with metadata"""
    content: str
    metadata: Dict
    embedding: np.ndarray = None


class SimpleRAG:
    """Simplified RAG system using Ollama"""

    def __init__(self, data_dir: str = "data"):
        """Initialize RAG components"""
        print("Initializing RAG System...")

        # Initialize components
        self.data_dir = data_dir
        self.chunk_size = 500
        self.chunk_overlap = 50

        # Load embedding model
        print("Loading BGE-large-en-v1.5 embedding model...")
        self.embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        self.embedding_dim = 1024

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.documents = []

        # Ollama configuration
        self.ollama_url = "http://localhost:11434"
        self.llm_model = "gemma3:4b-it-qat"  # Using Gemma3 4B model

    def check_ollama(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                if any(self.llm_model in name for name in model_names):
                    print(f"✓ Ollama is running with {self.llm_model}")
                    return True
                else:
                    print(f"⚠ Ollama is running but {self.llm_model} not found")
                    print(f"Available models: {', '.join(model_names)}")
                    print(f"\nPlease run: ollama pull {self.llm_model}")
                    return False
        except:
            print("⚠ Ollama is not running")
            print(f"Please start Ollama and run: ollama pull {self.llm_model}")
            return False

    def load_pdf(self, pdf_path: str) -> str:
        """Load and extract text from PDF"""
        text = ""
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n[Page {page_num + 1}]\n{page_text}"
        except Exception as e:
            print(f"Error loading {pdf_path}: {e}")
        return text

    def chunk_text(self, text: str, source: str) -> List[Document]:
        """Split text into overlapping chunks"""
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        chunks = []
        words = text.split()

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)

            if len(chunk_text) > 50:  # Minimum chunk size
                doc = Document(
                    content=chunk_text,
                    metadata={
                        'source': source,
                        'chunk_id': len(chunks),
                        'start_index': i
                    }
                )
                chunks.append(doc)

        return chunks

    def load_documents(self):
        """Load all PDF documents from data directory"""
        pdf_files = list(Path(self.data_dir).glob('*.pdf'))

        if not pdf_files:
            print(f"No PDF files found in {self.data_dir}")
            return

        print(f"\nFound {len(pdf_files)} PDF files")

        all_chunks = []
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file.name}...")
            text = self.load_pdf(str(pdf_file))

            if text:
                chunks = self.chunk_text(text, pdf_file.name)
                all_chunks.extend(chunks)
                print(f"  → Created {len(chunks)} chunks")

        if all_chunks:
            print(f"\nTotal chunks: {len(all_chunks)}")
            self.index_documents(all_chunks)

    def index_documents(self, documents: List[Document]):
        """Create embeddings and add to FAISS index"""
        print("\nGenerating embeddings...")

        # Extract texts
        texts = [doc.content for doc in documents]

        # Generate embeddings with progress bar
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        # Add to FAISS
        self.index.add(embeddings.astype('float32'))
        self.documents = documents

        print(f"✓ Indexed {len(documents)} documents")

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant documents"""
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        )

        # Search in FAISS
        scores, indices = self.index.search(
            query_embedding.astype('float32'),
            min(k, len(self.documents))
        )

        # Return documents
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                doc.score = float(score)
                results.append(doc)

        return results

    def generate_response(self, prompt: str) -> str:
        """Generate response using Ollama"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 512
                    }
                }
            )

            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error calling Ollama: {e}"

    def query(self, question: str, k: int = 3) -> Dict:
        """Main RAG query function"""
        # Search for relevant documents
        relevant_docs = self.search(question, k=k)

        if not relevant_docs:
            return {
                "question": question,
                "answer": "No relevant documents found.",
                "sources": []
            }

        # Build context from top documents
        context_parts = []
        sources = []

        for i, doc in enumerate(relevant_docs[:3], 1):
            context_parts.append(
                f"[Document {i} - {doc.metadata['source']}]\n{doc.content[:400]}..."
            )
            sources.append({
                "source": doc.metadata['source'],
                "score": getattr(doc, 'score', 0)
            })

        context = "\n\n".join(context_parts)

        # Create prompt for Gemma3
        prompt = f"""You are a helpful assistant. Answer the question based on the provided context.
If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {question}

Answer:"""

        # Generate response
        answer = self.generate_response(prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "context_used": context[:500] + "..."
        }

    def save_index(self, path: str = "rag_index"):
        """Save FAISS index and documents"""
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.faiss")

        # Save documents
        docs_data = []
        for doc in self.documents:
            docs_data.append({
                'content': doc.content,
                'metadata': doc.metadata
            })

        with open(f"{path}_docs.json", 'w') as f:
            json.dump(docs_data, f)

        print(f"✓ Index saved to {path}.faiss and {path}_docs.json")

    def load_index(self, path: str = "rag_index"):
        """Load FAISS index and documents"""
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.faiss")

        # Load documents
        with open(f"{path}_docs.json", 'r') as f:
            docs_data = json.load(f)

        self.documents = []
        for doc_data in docs_data:
            self.documents.append(Document(
                content=doc_data['content'],
                metadata=doc_data['metadata']
            ))

        print(f"✓ Loaded {len(self.documents)} documents from index")


def main():
    """Main function"""
    # Initialize RAG system
    rag = SimpleRAG(data_dir="data")

    # Check Ollama
    if not rag.check_ollama():
        print("\n⚠ Please ensure Ollama is running with Gemma3")
        print("Steps:")
        print("1. Install Ollama: https://ollama.ai")
        print("2. Start Ollama: ollama serve")
        print(f"3. Pull model: ollama pull {rag.llm_model}")
        return

    # Load documents
    print("\n" + "="*50)
    print("Loading documents...")
    print("="*50)
    rag.load_documents()

    if not rag.documents:
        print("No documents loaded. Exiting.")
        return

    # Interactive query loop
    print("\n" + "="*50)
    print("RAG System Ready!")
    print(f"Model: {rag.llm_model}")
    print(f"Documents: {len(rag.documents)} chunks")
    print("Type 'quit' to exit")
    print("="*50)

    while True:
        try:
            question = input("\n💭 Question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if question:
                print("\n🔍 Searching for relevant information...")
                result = rag.query(question)

                print("\n📚 Sources used:")
                for source in result['sources']:
                    print(f"  • {source['source']} (score: {source['score']:.3f})")

                print(f"\n💡 Answer:\n{result['answer']}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    # Check required packages
    import sys

    try:
        import faiss
        import pypdf
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print("Missing required packages!")
        print("\nPlease install:")
        print("pip install faiss-cpu sentence-transformers pypdf requests")
        sys.exit(1)

    main()