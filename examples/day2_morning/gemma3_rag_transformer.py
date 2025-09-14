#!/usr/bin/env python3
"""
RAG System using Gemma3 and BGE-large-en-v1.5
Database: FAISS
Data source: PDF files from examples/day2_morning/data/
"""

import os
import torch
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# PDF processing
try:
    from pypdf import PdfReader
except ImportError:
    import PyPDF2
    PdfReader = PyPDF2.PdfFileReader

# Embeddings and LLM
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Vector database
import faiss

# Text processing
import re
from dataclasses import dataclass


@dataclass
class Document:
    """Document chunk with metadata"""
    content: str
    metadata: Dict
    embedding: np.ndarray = None


class PDFProcessor:
    """Process PDF files into chunks"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_pdf(self, pdf_path: str) -> str:
        """Load PDF and extract text"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n[Page {page_num + 1}]\n{page_text}"
        except Exception as e:
            print(f"Error loading PDF {pdf_path}: {e}")
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

    def process_directory(self, directory: str) -> List[Document]:
        """Process all PDFs in directory"""
        all_documents = []
        pdf_files = Path(directory).glob('*.pdf')

        for pdf_file in pdf_files:
            print(f"Processing {pdf_file.name}...")
            text = self.load_pdf(str(pdf_file))

            if text:
                chunks = self.chunk_text(text, pdf_file.name)
                all_documents.extend(chunks)
                print(f"  Created {len(chunks)} chunks from {pdf_file.name}")

        return all_documents


class EmbeddingModel:
    """BGE-large-en-v1.5 embedding model"""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = 1024  # BGE-large-en-v1.5 dimension

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # BGE models need normalization
        )
        return embeddings


class FAISSIndex:
    """FAISS vector database"""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
        self.documents = []

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents with embeddings to index"""
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(documents)
        print(f"Added {len(documents)} documents to index")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))

        return results

    def save(self, path: str):
        """Save index to disk"""
        faiss.write_index(self.index, f"{path}.index")
        # Save documents separately (you'd need to implement serialization)
        print(f"Index saved to {path}.index")

    def load(self, path: str):
        """Load index from disk"""
        self.index = faiss.read_index(f"{path}.index")
        # Load documents (implement deserialization)
        print(f"Index loaded from {path}.index")


class Gemma3Generator:
    """Gemma3 model for text generation"""

    def __init__(self, model_name: str = "google/gemma-3-1b-it"):
        print(f"Loading Gemma3 model: {model_name}")

        # Using Google's Gemma3 1B instruction-tuned model
        # Other options: "google/gemma-3-4b-it", "google/gemma-3-12b-it"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate response using Gemma3"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from response
        response = response[len(prompt):].strip()

        return response


class RAGSystem:
    """Complete RAG system"""

    def __init__(self, use_ollama: bool = False):
        """
        Initialize RAG system
        Args:
            use_ollama: If True, use Ollama instead of loading model locally
        """
        self.pdf_processor = PDFProcessor(chunk_size=500, chunk_overlap=50)
        self.embedding_model = EmbeddingModel()
        self.vector_db = FAISSIndex(self.embedding_model.dimension)

        self.use_ollama = use_ollama
        if use_ollama:
            # Use Ollama for generation
            print("Using Ollama for generation")
            self.generator = None  # Will use Ollama API
        else:
            # Load Gemma3 locally using transformers
            self.generator = Gemma3Generator()

        self.documents = []

    def load_documents(self, directory: str):
        """Load and process all documents"""
        print(f"\nLoading documents from {directory}")
        self.documents = self.pdf_processor.process_directory(directory)
        print(f"Total documents loaded: {len(self.documents)}")

        if self.documents:
            # Generate embeddings
            print("\nGenerating embeddings...")
            texts = [doc.content for doc in self.documents]
            embeddings = self.embedding_model.encode(texts)

            # Add to vector database
            self.vector_db.add_documents(self.documents, embeddings)
            print("Documents indexed successfully!")

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents for query"""
        # Encode query
        query_embedding = self.embedding_model.encode([query])

        # Search
        results = self.vector_db.search(query_embedding[0], k=k)

        return [doc for doc, score in results]

    def generate_with_ollama(self, prompt: str) -> str:
        """Generate using Ollama (requires Ollama to be running)"""
        import requests
        import json

        url = "http://localhost:11434/api/generate"
        data = {
            "model": "gemma3:4b-it-qat",  # Or any other Ollama model
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                return response.json()['response']
            else:
                return "Error: Ollama not available. Please start Ollama"
        except:
            return "Error: Cannot connect to Ollama. Please ensure Ollama is running."

    def query(self, question: str, k: int = 5) -> str:
        """Main RAG query function"""
        print(f"\nQuery: {question}")

        # Retrieve relevant documents
        relevant_docs = self.retrieve(question, k=k)

        if not relevant_docs:
            return "No relevant documents found."

        # Build context
        context = "\n\n".join([
            f"[Source: {doc.metadata['source']}]\n{doc.content[:500]}..."
            for doc in relevant_docs[:3]  # Use top 3 for context
        ])

        # Build prompt
        prompt = f"""You are a helpful AI assistant. Use the following context to answer the question.
If you cannot answer based on the context, say so.

Context:
{context}

Question: {question}

Answer: """

        # Generate response
        if self.use_ollama:
            response = self.generate_with_ollama(prompt)
        else:
            response = self.generator.generate(prompt)

        return response


def main():
    """Main function to run RAG system"""

    # Configuration
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(script_dir, "data")  # Absolute path to data directory
    USE_OLLAMA = False  # Set to False to use transformers with Gemma3

    # Initialize RAG system
    print("Initializing RAG System...")
    rag = RAGSystem(use_ollama=USE_OLLAMA)

    # Load documents
    rag.load_documents(DATA_DIR)

    # Interactive query loop
    print("\n" + "="*50)
    print("RAG System Ready!")
    print("Type 'quit' to exit")
    print("="*50)

    while True:
        question = input("\nEnter your question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if question:
            answer = rag.query(question)
            print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    # Check if required packages are installed
    import importlib
    import sys

    missing_packages = []

    # Check each package with correct module names
    package_checks = {
        "torch": "torch",
        "transformers": "transformers",
        "sentence-transformers": "sentence_transformers",
        "faiss-cpu": "faiss",
        "pypdf": "pypdf",
        "numpy": "numpy"
    }

    for package_name, module_name in package_checks.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        print("Missing required packages:")
        print(f"Please install: pip install {' '.join(missing_packages)}")
        print("\nComplete installation command:")
        print("pip install torch transformers sentence-transformers faiss-cpu pypdf numpy")
        sys.exit(1)

    main()