# enhanced_retrieval.py
"""
Enhanced retrieval module for RAG system with fine-tuning capabilities
This module provides advanced retrieval techniques including:
1. Hybrid Dense+Sparse retrieval
2. Re-ranking with cross-encoders
3. Fine-tunable embeddings
4. Intelligent chunking
"""

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import spacy
import pickle
import os
from typing import List, Tuple, Optional
import json

class EnhancedRetriever:
    """Enhanced retrieval system with multiple retrieval strategies"""
    
    def __init__(self, 
                 embedding_model='all-MiniLM-L6-v2',
                 reranker_model='cross-encoder/ms-marco-MiniLM-L-6-v2',
                 use_hybrid=True,
                 use_reranking=True):
        
        self.embedding_model = SentenceTransformer(embedding_model)
        self.reranker = CrossEncoder(reranker_model) if use_reranking else None
        self.use_hybrid = use_hybrid
        self.use_reranking = use_reranking
        
        # Initialize components
        self.documents = []
        self.dense_embeddings = []
        self.bm25 = None
        
        # Load spacy for intelligent chunking
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def intelligent_chunk(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Create intelligent chunks based on semantic boundaries"""
        if not self.nlp:
            # Fallback to simple chunking
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]
        
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    
                    # Create overlap
                    overlap_count = min(len(current_chunk), overlap // 100)
                    if overlap_count > 0:
                        overlap_sentences = current_chunk[-overlap_count:]
                        current_chunk = overlap_sentences + [sentence]
                        current_length = sum(len(s) for s in current_chunk)
                    else:
                        current_chunk = [sentence]
                        current_length = sentence_length
                else:
                    # Single sentence longer than chunk_size - split it
                    words = sentence.split()
                    for i in range(0, len(words), chunk_size//10):
                        chunk_words = words[i:i+chunk_size//10]
                        chunks.append(' '.join(chunk_words))
                    current_chunk = []
                    current_length = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def add_documents(self, documents: List[str], chunk_size: int = 500):
        """Add documents to the retrieval system with intelligent chunking"""
        
        # Chunk all documents
        all_chunks = []
        for doc in documents:
            chunks = self.intelligent_chunk(doc, chunk_size)
            all_chunks.extend(chunks)
        
        self.documents = all_chunks
        
        # Create dense embeddings
        print(f"Creating embeddings for {len(all_chunks)} chunks...")
        self.dense_embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)
        
        # Create BM25 index for sparse retrieval
        if self.use_hybrid:
            tokenized_docs = [doc.split() for doc in all_chunks]
            self.bm25 = BM25Okapi(tokenized_docs)
        
        print(f"Successfully indexed {len(all_chunks)} document chunks")
    
    def dense_retrieve(self, query: str, k: int = 10) -> Tuple[List[int], List[float]]:
        """Dense retrieval using embeddings"""
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.dense_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:k]
        return top_indices.tolist(), similarities[top_indices].tolist()
    
    def hybrid_retrieve(self, query: str, k: int = 10, alpha: float = 0.7) -> Tuple[List[int], List[float]]:
        """Hybrid dense + sparse retrieval"""
        if not self.use_hybrid or self.bm25 is None:
            return self.dense_retrieve(query, k)
        
        # Dense scores
        query_embedding = self.embedding_model.encode([query])
        dense_scores = cosine_similarity(query_embedding, self.dense_embeddings)[0]
        
        # Sparse scores (BM25)
        sparse_scores = np.array(self.bm25.get_scores(query.split()))
        
        # Normalize scores to [0, 1]
        if np.max(dense_scores) > np.min(dense_scores):
            dense_scores = (dense_scores - np.min(dense_scores)) / (np.max(dense_scores) - np.min(dense_scores))
        if np.max(sparse_scores) > np.min(sparse_scores):
            sparse_scores = (sparse_scores - np.min(sparse_scores)) / (np.max(sparse_scores) - np.min(sparse_scores))
        
        # Combine scores
        combined_scores = alpha * dense_scores + (1 - alpha) * sparse_scores
        
        # Get top-k
        top_indices = np.argsort(combined_scores)[::-1][:k]
        return top_indices.tolist(), combined_scores[top_indices].tolist()
    
    def retrieve_with_reranking(self, query: str, initial_k: int = 20, final_k: int = 5) -> Tuple[List[int], List[float]]:
        """Retrieve with re-ranking using cross-encoder"""
        
        # Initial retrieval
        if self.use_hybrid:
            initial_indices, _ = self.hybrid_retrieve(query, initial_k)
        else:
            initial_indices, _ = self.dense_retrieve(query, initial_k)
        
        if not self.use_reranking or self.reranker is None:
            return initial_indices[:final_k], [1.0] * min(final_k, len(initial_indices))
        
        # Re-rank using cross-encoder
        candidate_pairs = [(query, self.documents[i]) for i in initial_indices]
        rerank_scores = self.reranker.predict(candidate_pairs)
        
        # Get final top-k
        final_indices_in_initial = np.argsort(rerank_scores)[::-1][:final_k]
        final_indices = [initial_indices[i] for i in final_indices_in_initial]
        final_scores = rerank_scores[final_indices_in_initial]
        
        return final_indices, final_scores.tolist()
    
    def search(self, query: str, k: int = 5, use_reranking: Optional[bool] = None) -> Tuple[List[str], List[float]]:
        """Main search interface"""
        if not self.documents:
            raise ValueError("No documents have been added to the retriever")
        
        use_rerank = use_reranking if use_reranking is not None else self.use_reranking
        
        if use_rerank:
            indices, scores = self.retrieve_with_reranking(query, initial_k=min(20, len(self.documents)), final_k=k)
        elif self.use_hybrid:
            indices, scores = self.hybrid_retrieve(query, k)
        else:
            indices, scores = self.dense_retrieve(query, k)
        
        retrieved_docs = [self.documents[i] for i in indices]
        return retrieved_docs, scores
    
    def fine_tune_embeddings(self, 
                           training_queries: List[str], 
                           relevant_docs: List[List[str]],
                           output_path: str = './fine-tuned-retriever',
                           epochs: int = 3):
        """Fine-tune the embedding model on domain-specific data"""
        
        # Create training examples
        train_examples = []
        
        for query, relevant_doc_list in zip(training_queries, relevant_docs):
            # Positive examples
            for doc in relevant_doc_list:
                train_examples.append(InputExample(texts=[query, doc], label=1.0))
            
            # Negative examples (sample from other documents)
            import random
            all_other_docs = [doc for docs in relevant_docs for doc in docs if doc not in relevant_doc_list]
            if all_other_docs:
                negative_samples = random.sample(all_other_docs, min(len(relevant_doc_list), len(all_other_docs)))
                for neg_doc in negative_samples:
                    train_examples.append(InputExample(texts=[query, neg_doc], label=0.0))
        
        # Create DataLoader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        
        # Define loss function
        train_loss = losses.CosineSimilarityLoss(self.embedding_model)
        
        # Fine-tune
        print(f"Fine-tuning embedding model with {len(train_examples)} examples...")
        self.embedding_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=100,
            output_path=output_path
        )
        
        print(f"Fine-tuned model saved to {output_path}")
        
        # Re-encode documents with fine-tuned model
        if self.documents:
            print("Re-encoding documents with fine-tuned model...")
            self.dense_embeddings = self.embedding_model.encode(self.documents, show_progress_bar=True)
    
    def save_index(self, filepath: str):
        """Save the retrieval index"""
        index_data = {
            'documents': self.documents,
            'dense_embeddings': self.dense_embeddings.tolist() if isinstance(self.dense_embeddings, np.ndarray) else self.dense_embeddings,
            'use_hybrid': self.use_hybrid,
            'use_reranking': self.use_reranking
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str):
        """Load a saved retrieval index"""
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)
        
        self.documents = index_data['documents']
        self.dense_embeddings = np.array(index_data['dense_embeddings'])
        self.use_hybrid = index_data.get('use_hybrid', True)
        self.use_reranking = index_data.get('use_reranking', True)
        
        # Recreate BM25 index if needed
        if self.use_hybrid and self.documents:
            tokenized_docs = [doc.split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
        
        print(f"Index loaded from {filepath} with {len(self.documents)} documents")

# Evaluation utilities
class RetrievalEvaluator:
    """Evaluate retrieval performance"""
    
    @staticmethod
    def calculate_recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate Recall@K"""
        if not relevant_docs:
            return 0.0
        
        retrieved_k = retrieved_docs[:k]
        relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant_docs)
        return relevant_retrieved / len(relevant_docs)
    
    @staticmethod
    def calculate_precision_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate Precision@K"""
        if not retrieved_docs:
            return 0.0
        
        retrieved_k = retrieved_docs[:k]
        relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant_docs)
        return relevant_retrieved / len(retrieved_k)
    
    @staticmethod
    def calculate_mrr(retrieved_docs_list: List[List[str]], relevant_docs_list: List[List[str]]) -> float:
        """Calculate Mean Reciprocal Rank (MRR)"""
        reciprocal_ranks = []
        
        for retrieved_docs, relevant_docs in zip(retrieved_docs_list, relevant_docs_list):
            for i, doc in enumerate(retrieved_docs):
                if doc in relevant_docs:
                    reciprocal_ranks.append(1.0 / (i + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


if __name__ == "__main__":
    # Example usage
    retriever = EnhancedRetriever(
        use_hybrid=True,
        use_reranking=True
    )
    
    # Add some sample documents
    sample_docs = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to learn patterns.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information."
    ]
    
    retriever.add_documents(sample_docs)
    
    # Search example
    results, scores = retriever.search("What is machine learning?", k=2)
    print("Search results:")
    for doc, score in zip(results, scores):
        print(f"Score: {score:.3f} | Document: {doc}")