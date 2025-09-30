import os
import chromadb
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")
       
        
        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def clear_collection(self):
        """
        Clear all documents from the collection by deleting and recreating it.
        """
        try:
            print(f"🗑️  Clearing collection: {self.collection_name}")
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document collection"},
            )
            print(f"✅ Collection cleared and recreated: {self.collection_name}")
        except Exception as e:
            print(f"⚠️  Error clearing collection: {e}")
            # If collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document collection"},
            )

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Simple text chunking by splitting on spaces and grouping into chunks.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
    )
        chunks = splitter.split_text(text)
    
        return chunks

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """
        print("VectorDB.add_documents: start")
        print(f"Processing {len(documents)} documents...")
        ids, texts, metadatas = [], [], []
        for doc_index, doc in enumerate(documents):
        # Handle both raw strings and dicts
            if isinstance(doc, str):
                text = doc
                metadata = {}
            else:
                text = doc.get("content", "")
                metadata = doc.get("metadata", {})
            #Split into chunks
            chunks = self.chunk_text(text, chunk_size=500)

            # Process each chunk
            for chunk_index, chunk in enumerate(chunks):
                chunk_id = f"doc_{doc_index}_chunk_{chunk_index}"

                ids.append(chunk_id)
                texts.append(chunk)
                metadatas.append({**metadata, "chunk_index": chunk_index})
            
                if chunk_index % 5 == 0:
                    print(f"Processed chunk {chunk_index} of document {doc_index}")
        
            # Embed chunks
        print(f"Prepared {len(texts)} chunks. Encoding embeddings...")
        embeddings = self.embedding_model.encode(texts,
        batch_size=16,          # smaller batches improve responsiveness
        show_progress_bar=True )
        
          # Ensure embeddings are list of flat floats
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

             # Flatten any accidental nested embeddings
        embeddings = [
            e[0].tolist() if isinstance(e, (list, np.ndarray)) and len(e) == 1 else e
            for e in embeddings
              ]
        
    
        print(f"Got embeddings shape: {len(embeddings)}x{len(embeddings[0]) if embeddings else 0}")
        print(f"About to call collection.add with {len(ids)} items")
        print("Sample embedding type:", type(embeddings[0]))
        print("Sample embedding length:", len(embeddings[0]))
        print("Sample embedding first 5 values:", embeddings[0][:5])
    
        # Add in batches
        try:
            batch_size = 100
            print(f"About to call collection.add with {len(ids)} items (batched)")
            for start in range(0, len(ids), batch_size):
                end = min(start + batch_size, len(ids))
                print(f"  → Adding batch {start} to {end}...")
                #sanity checks
                print("    lens:",
                      len(ids[start:end]),
                      len(embeddings[start:end]),
                      len(texts[start:end]),
                      len(metadatas[start:end]))
                
                print("    types:",
                      type(ids[start:end][0]),
                      type(embeddings[start:end][0]),
                      type(texts[start:end][0]),
                      type(metadatas[start:end][0]))
                
                if not all(isinstance(m, dict) for m in metadatas[start:end]):
                    print("⚠️ Bad metadata format in this batch, skipping...")
                    continue

                
                self.collection.add(
                     ids=ids[start:end],
                     embeddings=embeddings[start:end],
                     documents=texts[start:end],
                     metadatas=metadatas[start:end],
            )
                print(f"✅ Added batch {start}–{end}, collection count now: {self.collection.count()}")
            
            print("🎉 All documents added to vector database")
            print("Collection count:", self.collection.count())
        
        except Exception as e:
            print(f"Error in self.collection.add: {e}")
            raise

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        #Create query embedding
        query_embedding = self.embedding_model.encode([query])

        #Run similarity search in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),  # convert to list for ChromaDB
            n_results=n_results
    )

    # 3. Handle empty results
        if not results or "documents" not in results:
            return {
            "documents": [],
            "metadatas": [],
            "distances": [],
            "ids": [],

        
        }

        return {
        "documents": results["documents"][0],  # ChromaDB returns nested lists
        "metadatas": results["metadatas"][0],
        "distances": results["distances"][0],
        "ids": results["ids"][0],
        }
    
    