# ChromaDB Integration Example for RAG Application

"""
This file demonstrates how to integrate ChromaDB into the existing rag_app.py.
This is a reference implementation showing the minimal changes needed.

Note: This is an example/reference file. The actual rag_app.py has not been 
modified to maintain backward compatibility with the current pickle-based approach.
"""

import chromadb
from chromadb.config import Settings
import os
from typing import List, Tuple
import numpy as np

# Configuration
CHROMADB_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMADB_PORT = int(os.getenv("CHROMA_PORT", "8000"))
USE_CHROMADB = os.getenv("USE_CHROMADB", "false").lower() == "true"


class ChromaDBManager:
    """Manager class for ChromaDB operations"""
    
    def __init__(self, collection_name: str = "rag_documents"):
        """
        Initialize ChromaDB client and collection.
        
        Args:
            collection_name: Name of the collection to use
        """
        self.client = chromadb.HttpClient(
            host=CHROMADB_HOST,
            port=CHROMADB_PORT
        )
        self.collection_name = collection_name
        self.collection = None
    
    def create_or_get_collection(self):
        """Create or retrieve a collection"""
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "RAG application embeddings"}
            )
            return True
        except Exception as e:
            print(f"Error creating/getting collection: {e}")
            return False
    
    def add_embeddings(
        self, 
        chunks: List[str], 
        embeddings: List[List[float]], 
        metadata: dict = None
    ) -> bool:
        """
        Add embeddings to the collection.
        
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadata: Optional metadata dict to add to all chunks
            
        Returns:
            True if successful, False otherwise
        """
        if not self.collection:
            self.create_or_get_collection()
        
        try:
            # Generate IDs
            ids = [f"doc_{i}" for i in range(len(chunks))]
            
            # Prepare metadata
            metadatas = []
            for i in range(len(chunks)):
                meta = {"chunk_id": i, "chunk_length": len(chunks[i])}
                if metadata:
                    meta.update(metadata)
                metadatas.append(meta)
            
            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                end_idx = min(i + batch_size, len(chunks))
                self.collection.add(
                    embeddings=embeddings[i:end_idx],
                    documents=chunks[i:end_idx],
                    ids=ids[i:end_idx],
                    metadatas=metadatas[i:end_idx]
                )
            
            return True
        except Exception as e:
            print(f"Error adding embeddings: {e}")
            return False
    
    def query_similar(
        self, 
        query_embedding: List[float], 
        n_results: int = 3
    ) -> Tuple[List[str], List[float]]:
        """
        Query for similar documents.
        
        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            
        Returns:
            Tuple of (documents, distances)
        """
        if not self.collection:
            self.create_or_get_collection()
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # ChromaDB returns results in a specific format
            documents = results['documents'][0] if results['documents'] else []
            distances = results['distances'][0] if results['distances'] else []
            
            return documents, distances
        except Exception as e:
            print(f"Error querying collection: {e}")
            return [], []
    
    def get_collection_count(self) -> int:
        """Get number of items in collection"""
        if not self.collection:
            self.create_or_get_collection()
        
        try:
            return self.collection.count()
        except Exception as e:
            print(f"Error getting count: {e}")
            return 0
    
    def delete_collection(self):
        """Delete the collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = None
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
    
    def reset_collection(self):
        """Reset the collection (delete and recreate)"""
        self.delete_collection()
        return self.create_or_get_collection()


# Example integration functions that would replace existing functions in rag_app.py

def save_embeddings_chromadb(
    chunks: List[str], 
    embeddings: List[List[float]], 
    collection_name: str = "rag_documents"
) -> bool:
    """
    Save embeddings to ChromaDB (replaces save_embeddings function).
    
    This is equivalent to the existing pickle-based save_embeddings() function.
    """
    manager = ChromaDBManager(collection_name)
    if not manager.create_or_get_collection():
        return False
    
    return manager.add_embeddings(chunks, embeddings)


def load_embeddings_chromadb(
    collection_name: str = "rag_documents"
) -> Tuple[List[str], List[List[float]]]:
    """
    Load all embeddings from ChromaDB (replaces load_embeddings function).
    
    Note: This retrieves all documents. For large collections, 
    consider pagination or querying specific subsets.
    """
    manager = ChromaDBManager(collection_name)
    if not manager.create_or_get_collection():
        return [], []
    
    try:
        # Get all documents (for small collections)
        results = manager.collection.get()
        chunks = results['documents'] if results['documents'] else []
        embeddings = results['embeddings'] if results['embeddings'] else []
        
        return chunks, embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return [], []


def get_top_k_similar_docs_chromadb(
    query_vec: List[float], 
    collection_name: str = "rag_documents",
    k: int = 3
) -> Tuple[List[int], List[float]]:
    """
    Find top-k similar documents using ChromaDB (replaces get_top_k_similar_docs).
    
    Args:
        query_vec: Query embedding vector
        collection_name: Name of the collection to query
        k: Number of results to return
        
    Returns:
        Tuple of (indices, similarities)
        Note: ChromaDB returns distances, which we convert to similarities
    """
    manager = ChromaDBManager(collection_name)
    documents, distances = manager.query_similar(query_vec, k)
    
    # Convert distances to similarity scores (1 - normalized_distance)
    # ChromaDB uses L2 distance by default
    if distances:
        max_dist = max(distances) if max(distances) > 0 else 1
        similarities = [1 - (d / max_dist) for d in distances]
    else:
        similarities = []
    
    # Return indices and similarities
    # Note: ChromaDB doesn't directly return indices, so we use range
    indices = list(range(len(documents)))
    
    return indices, similarities, documents


# Example of how to update the main Streamlit app

def integrate_chromadb_in_streamlit():
    """
    Example showing how to integrate ChromaDB into the Streamlit app.
    
    Key changes needed in rag_app.py:
    """
    
    # 1. Add imports at the top of rag_app.py
    """
    import chromadb
    from chromadb_integration import ChromaDBManager, USE_CHROMADB
    """
    
    # 2. Initialize ChromaDB manager in main()
    """
    # In main() function, after initializing OpenAI client:
    chroma_manager = ChromaDBManager() if USE_CHROMADB else None
    """
    
    # 3. Replace save_embeddings call in the embed_button section
    """
    # Old code:
    # save_embeddings(chunks, embeddings)
    
    # New code:
    if USE_CHROMADB and chroma_manager:
        success = chroma_manager.add_embeddings(chunks, embeddings)
        if success:
            st.success(f"Saved {len(chunks)} chunks to ChromaDB")
    else:
        save_embeddings(chunks, embeddings)
        st.success(f"Saved embeddings to {EMBEDDINGS_FILE}")
    """
    
    # 4. Replace similarity search in the RAG section
    """
    # Old code:
    # top_indices, similarities = get_top_k_similar_docs(question_vec, st.session_state.doc_embeddings)
    
    # New code:
    if USE_CHROMADB and chroma_manager:
        documents, distances = chroma_manager.query_similar(question_vec, k=3)
        top_docs = documents
    else:
        top_indices, similarities = get_top_k_similar_docs(
            question_vec, 
            st.session_state.doc_embeddings
        )
        top_docs = [st.session_state.doc_chunks[i] for i in top_indices]
    """
    
    # 5. Add a toggle in the UI to switch between storage backends
    """
    # In the sidebar:
    storage_backend = st.sidebar.radio(
        "Storage Backend",
        ["Pickle Files", "ChromaDB"],
        help="Choose where to store embeddings"
    )
    USE_CHROMADB = (storage_backend == "ChromaDB")
    """


# Example: Testing ChromaDB connection

def test_chromadb_connection():
    """Test ChromaDB connection and basic operations"""
    try:
        manager = ChromaDBManager("test_collection")
        
        # Test connection
        print("Testing ChromaDB connection...")
        if not manager.create_or_get_collection():
            print("❌ Failed to connect to ChromaDB")
            return False
        
        print("✅ Successfully connected to ChromaDB")
        
        # Test add operation
        test_chunks = ["This is a test", "Another test chunk"]
        test_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        print("Testing add operation...")
        if manager.add_embeddings(test_chunks, test_embeddings):
            print("✅ Successfully added test embeddings")
        else:
            print("❌ Failed to add embeddings")
            return False
        
        # Test query operation
        print("Testing query operation...")
        query_embedding = [0.15, 0.25, 0.35]
        docs, distances = manager.query_similar(query_embedding, n_results=1)
        
        if docs:
            print(f"✅ Successfully queried. Found: {docs[0]}")
        else:
            print("❌ Query returned no results")
            return False
        
        # Cleanup
        print("Cleaning up test collection...")
        manager.delete_collection()
        print("✅ All tests passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


if __name__ == "__main__":
    print("ChromaDB Integration Example")
    print("=" * 50)
    print("\nThis is a reference implementation.")
    print("To use ChromaDB, set environment variable:")
    print("  USE_CHROMADB=true")
    print("\nTesting connection to ChromaDB...")
    print("=" * 50)
    
    test_chromadb_connection()
