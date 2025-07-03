# rag_app.py

import streamlit as st # GUI libray for data science 
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI
import pickle

from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

def fetch_text_from_url(url):
    """
    Fetch the text data from specific URL

    Input: URL link  
    Output: the text array

    Meaning:  
    
    """
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')
        return soup.get_text()
    except:
        return ""

def get_embedding(client, documents):
    """
    Embeds a list of documents using the provided embedding model.

    Input:
        model (SentenceTransformer): A model that converts text to embeddings.
        documents (List[str]): A list of sentences or paragraphs.

    Output:
        np.ndarray: 2D array where each row is the embedding of a document.

    Meaning:
    Transforms the list of text documents into numerical vectors that
    represent their meaning in high-dimensional space.
    """
    response = client.embeddings.create(
        input=documents,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def get_top_k_similar_docs(query_vec, doc_vecs, k=3):
    """
    Finds the top-k most similar documents based on cosine similarity.

    Input:
        query_vec (np.ndarray): A vector representation of the user query.
        doc_vecs (np.ndarray): A list of vector embeddings for documents.
        k (int): Number of top documents to return.

    Output:
        (List[int], List[float]):  
            - Indices of top-k most similar documents  
            - Corresponding similarity scores  

    Meaning:

    """
    similarities = cosine_similarity([query_vec], doc_vecs)[0]  # shape: (num_docs,)
    top_indices = np.argsort(similarities)[::-1][:k]  # Sort by descending similarity
    return top_indices, similarities

def main():
    load_dotenv()  # take environment variables
    
    st.title("🧠 Simple RAG with Web Content")

    # --- Section 1: Input URL & Embed ---
    st.header("1. Embed Website Content")
    url = st.text_input("Enter a website URL:")
    embed_button = st.button("Embed Content")
    embed_and_save_button = st.button("Embed Content and save to *.pkl file")
    load_pkl_File_button = st.button("Load the embedded *.pkl file")

    client = OpenAI(
    # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    if "doc_chunks" not in st.session_state:
        st.session_state.doc_chunks = []
        st.session_state.doc_embeddings = []
    
    if embed_button and url:
        text = fetch_text_from_url(url)
        text = text.replace("\n","")
        chunks = [text[i:i+100] for i in range(0, len(text), 100)]
        embeddings = [get_embedding(client,chunk) for chunk in chunks]
        
        st.session_state.doc_chunks = chunks
        st.session_state.doc_embeddings = embeddings

        st.success(f"Embedded {len(chunks)} chunks from the website.")

    if embed_and_save_button and url:
        text = fetch_text_from_url(url)
        text = text.replace("\n","")
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]

        embeddings = [get_embedding(client,chunk) for chunk in chunks]

        # Save both embeddings and chunks
        with open("embeddings.pkl", "wb") as f:
            pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)
        
        st.session_state.doc_chunks = chunks
        st.session_state.doc_embeddings = embeddings

        st.success(f"✅ Embedded {len(chunks)} chunks from the website. and saved to embeddings.pkl")

    if load_pkl_File_button:
        # Load from file
        with open("embeddings.pkl", "rb") as f:
            data = pickle.load(f)

        # Access chunks and embeddings
        loaded_chunks = data["chunks"]
        loaded_embeddings = data["embeddings"]

        st.session_state.doc_chunks = loaded_chunks
        st.session_state.doc_embeddings = loaded_embeddings

    #prompt
    prompt = st.chat_input("Say something")
    if prompt:
        st.write(f"User has sent the following prompt: {prompt}")
        if len(st.session_state.doc_embeddings) != 0:
            question_vec = get_embedding(client,prompt)
            top_indices, similarities  = get_top_k_similar_docs(question_vec, st.session_state.doc_embeddings)
            top_docs = [st.session_state.doc_chunks[i] for i in top_indices]
            # Construct a prompt
            context = "\n".join(top_docs)

            # prompt before the prompt
            # you need to setup the prompt structure that allow to response on purpose to the user
            prompt = f"""You are a helpful assistant. Use the following context to answer the question:

            Context:
            {context}

            Question: {prompt}
            Answer:"""

            completion = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are an assistant who is helping answer a questions. Please answer as if you are talking to a 8 years old children"},
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            st.write(f"Answer: {completion.choices[0].message.content}")
            #st.text_area("Answer:", completion.choices[0].message.content, height=400)
        else:
            st.warning("Please embedded a website first.")
# Ensures the script runs only when executed directly, not when imported
if __name__ == '__main__':
    main()