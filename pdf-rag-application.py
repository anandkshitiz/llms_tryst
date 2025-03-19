import os
import sys
import time
import json
import argparse
import requests
from PyPDF2 import PdfReader
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api"
MODEL_NAME = "llama3.2"  # Change this to match your Ollama model name

class Document:
    """Class representing a document chunk with its metadata"""
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

def extract_text_from_pdf(pdf_path: str) -> List[Tuple[str, Dict]]:
    """Extract text from PDF file along with metadata"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
    reader = PdfReader(pdf_path)
    texts = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():  # Only add non-empty pages
            metadata = {"page": i + 1, "source": os.path.basename(pdf_path)}
            texts.append((text, metadata))
            
    return texts

def chunk_text(texts: List[Tuple[str, Dict]], chunk_size: int = 1000, overlap: int = 200) -> List[Document]:
    """Split texts into chunks with specified size and overlap"""
    chunks = []
    
    for text, metadata in texts:
        # Split into smaller chunks if too long
        if len(text) <= chunk_size:
            chunks.append(Document(text, metadata))
        else:
            # Create overlapping chunks
            for i in range(0, len(text), chunk_size - overlap):
                chunk_text = text[i:i + chunk_size]
                if len(chunk_text) < 100:  # Skip very small chunks
                    continue
                    
                # Update metadata with chunk info
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk"] = i // (chunk_size - overlap) + 1
                chunk_metadata["chunk_start"] = i
                chunk_metadata["chunk_end"] = min(i + chunk_size, len(text))
                
                chunks.append(Document(chunk_text, chunk_metadata))
                
    return chunks

class OllamaEmbedder:
    """Class to create embeddings using Ollama"""
    def __init__(self, model: str = MODEL_NAME):
        self.model = model
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts"""
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings
            
    def embed_query(self, text: str) -> List[float]:
        """Create embedding for a single text"""
        try:
            response = requests.post(
                f"{OLLAMA_API_URL}/embeddings",
                json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
            result = response.json()
            return result["embedding"]
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Fallback to TF-IDF if embeddings fail
            return self._tfidf_fallback(text)
            
    def _tfidf_fallback(self, text: str) -> List[float]:
        """Create a simple TF-IDF vector as fallback"""
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text])
        dense_vector = tfidf_matrix.toarray()[0]
        # Normalize the vector
        norm = np.linalg.norm(dense_vector)
        if norm > 0:
            dense_vector = dense_vector / norm
        return dense_vector.tolist()

class TFIDFRetriever:
    """Simple TF-IDF based retriever as fallback"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.documents = []
        self.vectors = None
        
    def add_documents(self, documents: List[Document]):
        """Add documents to the retriever"""
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        self.vectors = self.vectorizer.fit_transform(texts)
        
    def get_relevant_documents(self, query: str, k: int = 3) -> List[Document]:
        """Get most relevant documents for a query"""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        # Get indices of top k most similar documents
        top_indices = similarities.argsort()[-k:][::-1]
        
        return [self.documents[i] for i in top_indices]

class VectorStoreRetriever:
    """Simple in-memory vector store"""
    def __init__(self):
        self.embeddings = []
        self.documents = []
        
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """Add documents and their embeddings"""
        self.documents = documents
        self.embeddings = embeddings
        
    def get_relevant_documents(self, query_embedding: List[float], k: int = 3) -> List[Document]:
        """Get most relevant documents for a query embedding"""
        # Calculate similarities
        similarities = []
        for emb in self.embeddings:
            similarity = self._cosine_similarity(query_embedding, emb)
            similarities.append(similarity)
            
        # Get indices of top k most similar documents
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        return [self.documents[i] for i in top_indices]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0
            
        return dot_product / (norm_a * norm_b)

class LlamaQuery:
    """Class to query Llama using Ollama API"""
    def __init__(self, model: str = MODEL_NAME):
        self.model = model
        
    def query(self, prompt: str, context: str = "", temperature: float = 0.7) -> str:
        """Send a query to Llama with context"""
        full_prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {prompt}

Answer the question based only on the provided context. If the context doesn't contain the information needed, say "I don't have enough information to answer that question."

Answer:"""

        try:
            response = requests.post(
                f"{OLLAMA_API_URL}/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "temperature": temperature,
                    "stream": False
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["response"]
        except Exception as e:
            return f"Error querying model: {e}"

class PDFQuerySystem:
    """Main class for RAG-based PDF query system"""
    def __init__(self, pdf_path: str, use_embeddings: bool = True):
        self.pdf_path = pdf_path
        self.use_embeddings = use_embeddings
        self.documents = []
        self.llama_query = LlamaQuery()
        
        # Initialize either embedding-based or TF-IDF retriever
        if use_embeddings:
            self.embedder = OllamaEmbedder()
            self.retriever = VectorStoreRetriever()
        else:
            self.retriever = TFIDFRetriever()
            
    def initialize(self):
        """Process PDF and initialize the system"""
        print(f"Processing PDF: {self.pdf_path}")
        texts = extract_text_from_pdf(self.pdf_path)
        print(f"Extracted {len(texts)} pages of text")
        
        self.documents = chunk_text(texts)
        print(f"Created {len(self.documents)} chunks")
        
        if self.use_embeddings:
            print("Creating embeddings (this may take a while)...")
            embeddings = self.embedder.embed_documents([doc.page_content for doc in self.documents])
            self.retriever.add_documents(self.documents, embeddings)
        else:
            print("Using TF-IDF for document retrieval...")
            self.retriever.add_documents(self.documents)
            
        print("System initialized and ready for queries!")
        
    def query(self, question: str, k: int = 3) -> str:
        """Query the system with a question"""
        if not self.documents:
            return "Please initialize the system first."
            
        try:
            # Get relevant documents
            if self.use_embeddings:
                query_embedding = self.embedder.embed_query(question)
                relevant_docs = self.retriever.get_relevant_documents(query_embedding, k)
            else:
                relevant_docs = self.retriever.get_relevant_documents(question, k)
                
            # Create context from relevant documents
            context = "\n\n".join([f"[Page {doc.metadata.get('page', 'unknown')}] {doc.page_content}" for doc in relevant_docs])
            
            # Query Llama with the context
            answer = self.llama_query.query(question, context)
            
            return answer
            
        except Exception as e:
            return f"Error processing query: {e}"

def main():
    parser = argparse.ArgumentParser(description="Query PDF documents using Llama RAG")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--no-embeddings", action="store_true", help="Use TF-IDF instead of embeddings")
    args = parser.parse_args()
    
    system = PDFQuerySystem(args.pdf_path, use_embeddings=not args.no_embeddings)
    system.initialize()
    
    print("\nYou can now query the PDF. Type 'exit' to quit.\n")
    
    while True:
        try:
            query = input("\nEnter your query: ")
            if query.lower() in ["exit", "quit"]:
                break
                
            start_time = time.time()
            answer = system.query(query)
            elapsed_time = time.time() - start_time
            
            print(f"\nAnswer (took {elapsed_time:.2f}s):\n{answer}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye!")

if __name__ == "__main__":
    main()
