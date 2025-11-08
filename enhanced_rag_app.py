# enhanced_rag_app.py
"""
Enhanced RAG application with fine-tuning capabilities
This version includes:
1. Advanced retrieval strategies (hybrid, re-ranking)
2. Intelligent chunking
3. Fine-tuning capabilities
4. Evaluation metrics
5. Enhanced UI for configuration
"""

import streamlit as st
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
from openai import OpenAI
import pickle
import pdfplumber
import io
import json
from datetime import datetime
import pandas as pd

# Import our enhanced retrieval module
from enhanced_retrieval import EnhancedRetriever, RetrievalEvaluator

# Constants
CHUNK_SIZE_DEFAULT = 500
EMBEDDINGS_FILE = "embeddings.pkl"
ENHANCED_INDEX_FILE = "enhanced_index.pkl"
TRAINING_DATA_FILE = "training_data.json"

def fetch_text_from_url(url):
    """Fetch text content from URL"""
    if not url.startswith("http://") and not url.startswith("https://"):
        return ""
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')
        return soup.get_text()
    except:
        return ""

def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF file"""
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def save_training_data(queries, relevant_docs, feedback_scores):
    """Save training data for future fine-tuning"""
    training_data = {
        'timestamp': datetime.now().isoformat(),
        'queries': queries,
        'relevant_docs': relevant_docs,
        'feedback_scores': feedback_scores
    }
    
    # Load existing data if it exists
    existing_data = []
    if os.path.exists(TRAINING_DATA_FILE):
        try:
            with open(TRAINING_DATA_FILE, 'r') as f:
                existing_data = json.load(f)
        except:
            existing_data = []
    
    existing_data.append(training_data)
    
    with open(TRAINING_DATA_FILE, 'w') as f:
        json.dump(existing_data, f, indent=2)

def load_training_data():
    """Load saved training data"""
    if os.path.exists(TRAINING_DATA_FILE):
        try:
            with open(TRAINING_DATA_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def main():
    load_dotenv()
    
    st.set_page_config(
        page_title="Enhanced RAG System", 
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Enhanced RAG with Fine-tuning Capabilities")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Retrieval settings
        st.subheader("Retrieval Settings")
        use_hybrid = st.checkbox("Use Hybrid Retrieval (Dense + Sparse)", value=True)
        use_reranking = st.checkbox("Use Re-ranking", value=True)
        chunk_size = st.slider("Chunk Size", min_value=200, max_value=1000, value=500, step=50)
        alpha = st.slider("Hybrid Alpha (Dense weight)", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        
        # Model settings
        st.subheader("Model Settings")
        embedding_model = st.selectbox("Embedding Model", [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2", 
            "multi-qa-MiniLM-L6-cos-v1",
            "sentence-transformers/all-MiniLM-L12-v2"
        ])
        
        reranker_model = st.selectbox("Re-ranker Model", [
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "cross-encoder/ms-marco-TinyBERT-L-2-v2"
        ])
    
    # Check OpenAI API key
   # Load environment variables
    # This is useful for storing sensitive information like API keys.
    
    # Clear any existing OPENAI_API_KEY from system environment to ensure we use the file
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']
    
    # Now load from sample.env file
    load_dotenv()  # Load from sample.env file specifically
    openai_api_key = os.environ.get("OPENAI_API_KEY")    
     # Check if the OpenAI API key is set
    if not openai_api_key:
        st.error("OPENAI_API_KEY is not set. Please set it in your environment variables or Streamlit secrets.")
        st.stop()

    client = OpenAI(api_key=openai_api_key)
    
    # Initialize enhanced retriever
    if 'enhanced_retriever' not in st.session_state:
        st.session_state.enhanced_retriever = EnhancedRetriever(
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            use_hybrid=use_hybrid,
            use_reranking=use_reranking
        )
    
    # Initialize session state for training data
    if 'training_queries' not in st.session_state:
        st.session_state.training_queries = []
    if 'training_docs' not in st.session_state:
        st.session_state.training_docs = []
    if 'feedback_scores' not in st.session_state:
        st.session_state.feedback_scores = []
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 Document Processing", 
        "🔍 Enhanced Retrieval", 
        "💬 RAG Chat",
        "🎯 Fine-tuning",
        "📊 Evaluation"
    ])
    
    # Tab 1: Document Processing
    with tab1:
        st.header("Document Processing & Indexing")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            source_type = st.radio("Select content source:", 
                                 ["Web URL", "Text File", "PDF", "Enhanced Index File"])
            
            documents = []
            
            if source_type == "Web URL":
                urls = st.text_area("Enter URLs (one per line):", 
                                  placeholder="https://example.com\\nhttps://another.com")
                if urls:
                    url_list = [u.strip() for u in urls.splitlines() if u.strip()]
                    for url in url_list:
                        text = fetch_text_from_url(url)
                        if text:
                            documents.append(text.replace("\\n", " "))
            
            elif source_type == "Text File":
                uploaded_files = st.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True)
                for uploaded_file in uploaded_files:
                    text = uploaded_file.read().decode("utf-8")
                    documents.append(text.replace("\\n", " "))
            
            elif source_type == "PDF":
                uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
                for uploaded_file in uploaded_files:
                    text = extract_text_from_pdf(uploaded_file)
                    if text:
                        documents.append(text.replace("\\n", " "))
            
            elif source_type == "Enhanced Index File":
                uploaded_file = st.file_uploader("Upload enhanced index file", type=["pkl"])
                if uploaded_file:
                    try:
                        st.session_state.enhanced_retriever.load_index(uploaded_file)
                        st.success("Enhanced index loaded successfully!")
                    except Exception as e:
                        st.error(f"Failed to load index: {e}")
        
        with col2:
            st.subheader("Processing Status")
            if hasattr(st.session_state.enhanced_retriever, 'documents') and st.session_state.enhanced_retriever.documents:
                st.metric("Indexed Documents", len(st.session_state.enhanced_retriever.documents))
                st.success("✅ Index Ready")
            else:
                st.warning("⏳ No documents indexed")
        
        # Process documents
        if documents and st.button("🚀 Process Documents", type="primary"):
            with st.spinner("Processing documents with enhanced retrieval..."):
                try:
                    st.session_state.enhanced_retriever.add_documents(documents, chunk_size=chunk_size)
                    st.success(f"✅ Successfully processed {len(documents)} documents into {len(st.session_state.enhanced_retriever.documents)} chunks")
                    
                    # Update retriever settings
                    st.session_state.enhanced_retriever.use_hybrid = use_hybrid
                    st.session_state.enhanced_retriever.use_reranking = use_reranking
                    
                except Exception as e:
                    st.error(f"Error processing documents: {e}")
        
        # Save/Download index
        if hasattr(st.session_state.enhanced_retriever, 'documents') and st.session_state.enhanced_retriever.documents:
            st.subheader("💾 Save Index")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Enhanced Index"):
                    st.session_state.enhanced_retriever.save_index(ENHANCED_INDEX_FILE)
                    st.success("Index saved locally!")
            
            with col2:
                # Create download buffer
                buffer = io.BytesIO()
                st.session_state.enhanced_retriever.save_index("temp_index.pkl")
                with open("temp_index.pkl", "rb") as f:
                    buffer.write(f.read())
                buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Index",
                    data=buffer,
                    file_name=f"enhanced_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                    mime="application/octet-stream"
                )
    
    # Tab 2: Enhanced Retrieval Testing
    with tab2:
        st.header("🔍 Enhanced Retrieval Testing")
        
        if not hasattr(st.session_state.enhanced_retriever, 'documents') or not st.session_state.enhanced_retriever.documents:
            st.warning("Please process documents first in the Document Processing tab.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                test_query = st.text_input("Enter a test query:", placeholder="What is machine learning?")
                k_results = st.slider("Number of results", min_value=1, max_value=20, value=5)
                
                retrieval_method = st.radio("Retrieval Method:", [
                    "Hybrid (Dense + Sparse)",
                    "Dense Only", 
                    "With Re-ranking"
                ])
            
            with col2:
                st.subheader("Retrieval Settings")
                st.write(f"📊 Total chunks: {len(st.session_state.enhanced_retriever.documents)}")
                st.write(f"🔧 Chunk size: {chunk_size}")
                st.write(f"⚖️ Hybrid alpha: {alpha}")
            
            if test_query and st.button("🔍 Search", type="primary"):
                with st.spinner("Searching..."):
                    try:
                        if retrieval_method == "Dense Only":
                            indices, scores = st.session_state.enhanced_retriever.dense_retrieve(test_query, k_results)
                            results = [st.session_state.enhanced_retriever.documents[i] for i in indices]
                        elif retrieval_method == "Hybrid (Dense + Sparse)":
                            indices, scores = st.session_state.enhanced_retriever.hybrid_retrieve(test_query, k_results, alpha)
                            results = [st.session_state.enhanced_retriever.documents[i] for i in indices]
                        else:  # With Re-ranking
                            results, scores = st.session_state.enhanced_retriever.search(test_query, k_results, use_reranking=True)
                        
                        st.subheader("🎯 Search Results")
                        for i, (result, score) in enumerate(zip(results, scores)):
                            with st.expander(f"Result {i+1} (Score: {score:.4f})"):
                                st.write(result)
                    
                    except Exception as e:
                        st.error(f"Search error: {e}")
    
    # Tab 3: RAG Chat
    with tab3:
        st.header("💬 RAG-powered Chat")
        
        if not hasattr(st.session_state.enhanced_retriever, 'documents') or not st.session_state.enhanced_retriever.documents:
            st.warning("Please process documents first in the Document Processing tab.")
        else:
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant" and "retrieved_docs" in message:
                        with st.expander("📚 Retrieved Context"):
                            for i, doc in enumerate(message["retrieved_docs"]):
                                st.markdown(f"**Chunk {i+1}:** {doc[:200]}...")
            
            # Chat input
            if prompt := st.chat_input("Ask a question about your documents"):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            # Retrieve relevant documents
                            retrieved_docs, scores = st.session_state.enhanced_retriever.search(prompt, k=3)
                            context = "\\n\\n".join(retrieved_docs)
                            
                            # Generate response with OpenAI
                            prompt_text = f"""You are a helpful assistant. Use the following context to answer the question. If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {prompt}
Answer:"""
                            
                            completion = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                                    {"role": "user", "content": prompt_text},
                                ],
                                temperature=0.1
                            )
                            
                            response = completion.choices[0].message.content
                            st.markdown(response)
                            
                            # Show retrieved context
                            with st.expander("📚 Retrieved Context"):
                                for i, (doc, score) in enumerate(zip(retrieved_docs, scores)):
                                    st.markdown(f"**Chunk {i+1} (Score: {score:.3f}):** {doc[:200]}...")
                            
                            # Add feedback mechanism
                            col1, col2 = st.columns([3, 1])
                            with col2:
                                feedback = st.radio("Rate this response:", 
                                                  ["👍 Helpful", "👎 Not Helpful"], 
                                                  key=f"feedback_{len(st.session_state.messages)}")
                                if st.button("Submit Feedback", key=f"submit_{len(st.session_state.messages)}"):
                                    # Save training data
                                    st.session_state.training_queries.append(prompt)
                                    st.session_state.training_docs.append(retrieved_docs)
                                    st.session_state.feedback_scores.append(1 if feedback == "👍 Helpful" else 0)
                                    st.success("Feedback saved for fine-tuning!")
                            
                            # Add assistant message
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response,
                                "retrieved_docs": retrieved_docs
                            })
                            
                        except Exception as e:
                            st.error(f"Error generating response: {e}")
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.experimental_rerun()
    
    # Tab 4: Fine-tuning
    with tab4:
        st.header("🎯 Fine-tuning & Training")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Training Data")
            
            # Show collected training data
            if st.session_state.training_queries:
                df = pd.DataFrame({
                    'Query': st.session_state.training_queries,
                    'Feedback': ['👍' if score == 1 else '👎' for score in st.session_state.feedback_scores],
                    'Num_Retrieved_Docs': [len(docs) for docs in st.session_state.training_docs]
                })
                st.dataframe(df)
                
                # Manual training data input
                st.subheader("Add Manual Training Data")
                with st.expander("➕ Add Query-Document Pairs"):
                    manual_query = st.text_input("Training Query:")
                    manual_docs = st.text_area("Relevant Documents (one per line):")
                    
                    if st.button("Add Training Example") and manual_query and manual_docs:
                        doc_list = [d.strip() for d in manual_docs.split('\\n') if d.strip()]
                        st.session_state.training_queries.append(manual_query)
                        st.session_state.training_docs.append(doc_list)
                        st.session_state.feedback_scores.append(1)
                        st.success("Training example added!")
                        st.experimental_rerun()
            else:
                st.info("No training data collected yet. Use the RAG Chat to generate training data through user feedback.")
        
        with col2:
            st.subheader("Fine-tuning Actions")
            
            if st.session_state.training_queries:
                st.metric("Training Examples", len(st.session_state.training_queries))
                
                # Fine-tuning parameters
                epochs = st.slider("Training Epochs", min_value=1, max_value=10, value=3)
                
                if st.button("🚀 Start Fine-tuning", type="primary"):
                    with st.spinner("Fine-tuning embedding model..."):
                        try:
                            # Prepare training data
                            queries = st.session_state.training_queries
                            relevant_docs = st.session_state.training_docs
                            
                            # Start fine-tuning
                            st.session_state.enhanced_retriever.fine_tune_embeddings(
                                training_queries=queries,
                                relevant_docs=relevant_docs,
                                epochs=epochs
                            )
                            
                            st.success("✅ Fine-tuning completed!")
                            st.info("The model has been fine-tuned and document embeddings have been updated.")
                            
                        except Exception as e:
                            st.error(f"Fine-tuning error: {e}")
                
                # Save/Load training data
                if st.button("💾 Save Training Data"):
                    save_training_data(
                        st.session_state.training_queries,
                        st.session_state.training_docs, 
                        st.session_state.feedback_scores
                    )
                    st.success("Training data saved!")
                
                if st.button("📁 Load Training Data"):
                    loaded_data = load_training_data()
                    if loaded_data:
                        # Combine all training sessions
                        all_queries = []
                        all_docs = []
                        all_scores = []
                        for session in loaded_data:
                            all_queries.extend(session['queries'])
                            all_docs.extend(session['relevant_docs'])
                            all_scores.extend(session['feedback_scores'])
                        
                        st.session_state.training_queries = all_queries
                        st.session_state.training_docs = all_docs
                        st.session_state.feedback_scores = all_scores
                        st.success(f"Loaded {len(all_queries)} training examples!")
                        st.experimental_rerun()
    
    # Tab 5: Evaluation
    with tab5:
        st.header("📊 Retrieval Evaluation")
        
        if not hasattr(st.session_state.enhanced_retriever, 'documents') or not st.session_state.enhanced_retriever.documents:
            st.warning("Please process documents first.")
        else:
            st.subheader("Evaluation Metrics")
            
            # Test queries for evaluation
            test_queries = st.text_area("Enter test queries (one per line):", 
                                      placeholder="What is machine learning?\\nHow does AI work?")
            
            if test_queries:
                query_list = [q.strip() for q in test_queries.split('\\n') if q.strip()]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    eval_k = st.slider("Evaluate at K", min_value=1, max_value=10, value=5)
                with col2:
                    comparison_mode = st.checkbox("Compare Methods")
                with col3:
                    if st.button("🔍 Run Evaluation"):
                        evaluation_results = {}
                        
                        methods = ["Dense", "Hybrid", "With Re-ranking"] if comparison_mode else ["Current Settings"]
                        
                        for method in methods:
                            method_results = []
                            
                            for query in query_list:
                                if method == "Dense":
                                    indices, scores = st.session_state.enhanced_retriever.dense_retrieve(query, eval_k)
                                    results = [st.session_state.enhanced_retriever.documents[i] for i in indices]
                                elif method == "Hybrid":
                                    indices, scores = st.session_state.enhanced_retriever.hybrid_retrieve(query, eval_k)
                                    results = [st.session_state.enhanced_retriever.documents[i] for i in indices]
                                elif method == "With Re-ranking":
                                    results, scores = st.session_state.enhanced_retriever.search(query, eval_k, use_reranking=True)
                                else:  # Current settings
                                    results, scores = st.session_state.enhanced_retriever.search(query, eval_k)
                                
                                method_results.append({
                                    'query': query,
                                    'results': results,
                                    'scores': scores
                                })
                            
                            evaluation_results[method] = method_results
                        
                        # Display results
                        st.subheader("📈 Evaluation Results")
                        
                        if comparison_mode:
                            # Create comparison table
                            comparison_data = []
                            for method, results in evaluation_results.items():
                                avg_score = np.mean([np.mean(r['scores']) for r in results])
                                comparison_data.append({
                                    'Method': method,
                                    'Avg Score': f"{avg_score:.4f}",
                                    'Queries Tested': len(results)
                                })
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df)
                        
                        # Detailed results
                        for method, results in evaluation_results.items():
                            with st.expander(f"📊 {method} Results"):
                                for result in results:
                                    st.write(f"**Query:** {result['query']}")
                                    for i, (doc, score) in enumerate(zip(result['results'], result['scores'])):
                                        st.write(f"  {i+1}. (Score: {score:.4f}) {doc[:100]}...")
                                    st.write("---")

if __name__ == '__main__':
    main()