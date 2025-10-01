import os
import faiss
import json
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorDB:
    """
    A simple FAISS-based vector database wrapper with persistence.
    """

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 persist_dir: str = "./faiss_db"):
        # Load embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Paths for persistence
        self.persist_dir = persist_dir
        self.index_path = os.path.join(persist_dir, "index.faiss")
        self.meta_path = os.path.join(persist_dir, "meta.json")

        # FAISS index
        self.index = None
        self.dimension = None

        # Storage
        self.ids = []
        self.documents = []
        self.metadatas = []

        # Try to load persisted data
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self._load()
        else:
            print("No existing FAISS index found, starting fresh.")

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(text)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        print("FAISSVectorDB.add_documents: start")
        ids, texts, metadatas = [], [], []

        for doc_index, doc in enumerate(documents):
            if isinstance(doc, str):
                text, metadata = doc, {}
            else:
                text = doc.get("content", "")
                metadata = doc.get("metadata", {})

            chunks = self.chunk_text(text, chunk_size=500)
            for chunk_index, chunk in enumerate(chunks):
                chunk_id = f"doc_{doc_index}_chunk_{chunk_index}"
                ids.append(chunk_id)
                texts.append(chunk)
                metadatas.append({**metadata, "chunk_index": chunk_index})

        print(f"Prepared {len(texts)} chunks. Encoding embeddings...")
        embeddings = self.embedding_model.encode(texts, batch_size=16, show_progress_bar=True)

        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.astype("float32")

        if self.index is None:
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)
            print(f"Initialized FAISS index with dimension {self.dimension}")

        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        self.ids.extend(ids)
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)

        print(f"✅ Added {len(ids)} chunks to FAISS. Total size: {len(self.ids)}")

        # Save after adding
        self._save()

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        query_embedding = self.embedding_model.encode([query]).astype("float32")
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, n_results)

        results = {
            "ids": [self.ids[i] for i in indices[0]],
            "documents": [self.documents[i] for i in indices[0]],
            "metadatas": [self.metadatas[i] for i in indices[0]],
            "distances": distances[0].tolist(),
        }
        return results

    def _save(self):
        """Save FAISS index and metadata to disk."""
        os.makedirs(self.persist_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_path)

        meta = {
            "ids": self.ids,
            "documents": self.documents,
            "metadatas": self.metadatas,
            "dimension": self.dimension,
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"💾 Saved FAISS index and metadata to {self.persist_dir}")

    def _load(self):
        """Load FAISS index and metadata from disk."""
        print(f"📂 Loading FAISS index from {self.persist_dir}")
        self.index = faiss.read_index(self.index_path)

        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.ids = meta["ids"]
        self.documents = meta["documents"]
        self.metadatas = meta["metadatas"]
        self.dimension = meta["dimension"]

        print(f"✅ Loaded FAISS index with {len(self.ids)} chunks, dim={self.dimension}")
