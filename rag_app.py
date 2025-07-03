# rag_app.py

import streamlit as st # GUI libray for data science 
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
#from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI
import pickle
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# --- Constants ---
CHUNK_SIZE_EMBED = 100
CHUNK_SIZE_SAVE = 500
EMBEDDINGS_FILE = "embeddings.pkl"


def fetch_text_from_url(url):
    """
    Fetch the text data from specific URL using the requests library and BeautifulSoup.

    Input: URL link  
    Output: the text array

    Meaning:  
    in order to extract the text content from a webpage, we use the requests library to fetch the HTML content and BeautifulSoup to parse it.
    The function returns the text content of the webpage, which can then be used for further processing
    """
    # if the URL is not valid, return an empty string
    if not url.startswith("http://") and not url.startswith("https://"):
        return ""
    # try to fetch the content from the URL
    # if the URL is not valid or the content cannot be fetched, return an empty string
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')
        return soup.get_text()
    except:
        return ""

def get_embedding(client, documents):
    """
    get_embedding(client, documents) -> np.ndarray:
    Encodes the input documents into vector embeddings using the OpenAI API.

    Input:
        client (OpenAI): An instance of the OpenAI client to interact with the API.
        documents (List[str]): A list of documents to be embedded.

    Output:
        np.ndarray: The embedding vector for the first document in the list.

    Meaning:
    encodes the input documents into vector embeddings using the OpenAI API.
    The function takes a list of documents and returns the embedding for the first document. The model used is "text-embedding-3-small", which is suitable for generating embeddings for text data.
    The embeddings can be used for various tasks such as similarity search, clustering, or classification.
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
    This function computes the cosine similarity between a query vector and a list of document vectors, returning the indices and scores of the top-k most similar documents.
    The cosine similarity is a measure of similarity between two non-zero vectors, defined as the cosine of the angle between them. It is often used in information retrieval and natural language processing to find documents that are most similar to a given query.
    The function returns the indices of the top-k most similar documents and their corresponding similarity scores.
    """
    similarities = cosine_similarity([query_vec], doc_vecs)[0]  # shape: (num_docs,)
    top_indices = np.argsort(similarities)[::-1][:k]  # Sort by descending similarity
    return top_indices, similarities



# Initialize session state to store document chunks and embeddings
@st.cache_data
def initialize_session_state():
    """Initializes the session state for storing document chunks and embeddings.
    Meaning:
    This function checks if the session state variables for document chunks and embeddings exist, and if not, initializes them as empty lists.
    This is useful for maintaining state across user interactions in a Streamlit application, allowing the application to remember previously embedded documents and their corresponding embeddings.
    """
    if "doc_chunks" not in st.session_state:
        st.session_state.doc_chunks = []
        st.session_state.doc_embeddings = []

# Function to chunk text into smaller pieces
# This function takes a string and splits it into chunks of a specified size.
def chunk_text(text: str, chunk_size: int):
    """    Splits the input text into smaller chunks of a specified size.
    Input:
        text (str): The input text to be chunked.
        chunk_size (int): The size of each chunk.  
    Output:
        List[str]: A list of text chunks, each of size chunk_size or smaller.   
    Meaning:
    This function takes a string and splits it into smaller pieces of a specified size. It returns a list of text chunks, each of size chunk_size or smaller. This is useful for processing large texts in manageable pieces, especially when working with models that have input size limitations.
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Function to embed chunks of text using OpenAI API
# This function takes a list of text chunks and returns their embeddings using the OpenAI API.
def embed_chunks(client, chunks):
    """    Embeds a list of text chunks using the OpenAI API.
    Input:
        client (OpenAI): An instance of the OpenAI client to interact with the API.
        chunks (List[str]): A list of text chunks to be embedded. 
    Output:
        List[np.ndarray]: A list of embeddings for each text chunk.
    Meaning:
    This function takes a list of text chunks and returns their embeddings using the OpenAI API.
    The embeddings are vector representations of the text chunks, which can be used for various tasks such as similarity search, clustering, or classification.
    """
    return [get_embedding(client, chunk) for chunk in chunks]

# Function to save and load embeddings and chunks
# This function saves the embeddings and chunks to a file using pickle.
def save_embeddings(chunks, embeddings, filename=EMBEDDINGS_FILE):
    """    Saves the embeddings and chunks to a file using pickle.
    Input:
        chunks (List[str]): A list of text chunks.
        embeddings (List[np.ndarray]): A list of embeddings corresponding to the text chunks.
    filename (str): The name of the file to save the embeddings and chunks. Default is "embeddings.pkl".
    Meaning:
    This function saves the embeddings and chunks to a file using pickle. It allows you to persist the embeddings and chunks for later use, avoiding the need to re-embed the text every time the application is run.
    """
    with open(filename, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)

# Function to load embeddings and chunks from a file
# This function loads the embeddings and chunks from a file using pickle.
def load_embeddings(filename=EMBEDDINGS_FILE):
    """    Loads the embeddings and chunks from a file using pickle.
    Input:
        filename (str): The name of the file to load the embeddings and chunks from. Default is "embeddings.pkl".
    Output:
        Tuple[List[str], List[np.ndarray]]: A tuple containing a list of text chunks and a list of embeddings.
    Meaning:
    This function loads the embeddings and chunks from a file using pickle. It allows you to retrieve previously saved embeddings and chunks, enabling you to use them without re-embedding the text.
    """
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["chunks"], data["embeddings"]

# Main function to run the Streamlit app
def main():
    """Main function to run the Streamlit app for embedding and querying website content.
    Meaning:
    This function initializes the Streamlit app, sets up the user interface, and handles user interactions for embedding website content and querying it.
    It allows users to input a URL, embed the content from that URL, and then ask questions based on the embedded content.
    """
    # Load environment variables
    # This is useful for storing sensitive information like API keys.
    load_dotenv()  # take environment variables
    
    st.title("🧠 Simple RAG with Web Content")

    # --- Section 1: Input URL & Embed ---
    st.header("1. Embed Website Content")
    url = st.text_input("Enter a website URL:")
    embed_button = st.button("Embed Content")
    embed_and_save_button = st.button("Embed Content and save to *.pkl file")
    load_pkl_File_button = st.button("Load the embedded *.pkl file")

    # Initialize OpenAI client
    # This is the client that will be used to interact with the OpenAI API for embedding
    client = OpenAI(
    # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Initialize session state for document chunks and embeddings
    initialize_session_state()
    
    # --- Section 2: Display Embedded Content ---
    if embed_button and url:
        text = fetch_text_from_url(url).replace("\n", "")
        chunks = chunk_text(text, CHUNK_SIZE_EMBED)
        print(f"Number of chunks: {len(chunks)}")
        if len(chunks) == 0:
            st.error("No content found at the provided URL.")
            return
        
        embeddings = embed_chunks(client, chunks)
        if len(embeddings) == 0:
            st.error("Failed to embed the content.")
            return
        print(f"Number of embeddings: {len(embeddings)}")
        # Store chunks and embeddings in session state
        st.session_state.doc_chunks = chunks
        st.session_state.doc_embeddings = embeddings
        st.success(f"Embedded {len(chunks)} chunks from the website.")
    
    if embed_and_save_button and url:
        text = fetch_text_from_url(url).replace("\n", "")
        if not text:
            st.error("No content found at the provided URL.")
            return
        st.write(f"Fetched text from {url} with length: {len(text)} characters.")
        if len(text) < CHUNK_SIZE_SAVE:
            st.error(f"Text is too short to chunk. Minimum length is {CHUNK_SIZE_SAVE} characters.")
            return
        chunks = chunk_text(text, CHUNK_SIZE_SAVE)
        print(f"Number of chunks: {len(chunks)}")
        if len(chunks) == 0:
            st.error("No content found at the provided URL.")
            return
        embeddings = embed_chunks(client, chunks)
        if len(embeddings) == 0:
            st.error("Failed to embed the content.")
            return
        print(f"Number of embeddings: {len(embeddings)}")
        # Store chunks and embeddings in session state and save to file
        save_embeddings(chunks, embeddings)
        st.session_state.doc_chunks = chunks
        st.session_state.doc_embeddings = embeddings
        st.success(f"✅ Embedded {len(chunks)} chunks and saved to {EMBEDDINGS_FILE}")
    
    if load_pkl_File_button:
        try:
            chunks, embeddings = load_embeddings()
            st.session_state.doc_chunks = chunks
            st.session_state.doc_embeddings = embeddings
            st.success("Loaded embeddings from file.")
        except Exception as e:
            st.error(f"Failed to load embeddings: {e}")

    
    prompt = st.chat_input("Say something")
    if prompt:
        st.write(f"User has sent the following prompt: {prompt}")
        #check if embeddings are available
        if not st.session_state.doc_embeddings:
            st.warning("Please embed a website first.")
            return
        if st.session_state.doc_embeddings:
            question_vec = get_embedding(client, prompt)
            top_indices, similarities = get_top_k_similar_docs(question_vec, st.session_state.doc_embeddings)
            top_docs = [st.session_state.doc_chunks[i] for i in top_indices]
            context = "\n".join(top_docs)
            prompt_text = f"""You are a helpful assistant. Use the following context to answer the question:

            Context:
            {context}

            Question: {prompt}
            Answer:"""
            completion = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are an assistant who is helping answer a questions. Please answer as if you are talking to a 8 years old children"},
                    {"role": "user", "content": prompt_text},
                ],
            )
            st.write(f"Answer: {completion.choices[0].message.content}")
        else:
            st.warning("Please embed a website first.")
# Ensures the script runs only when executed directly, not when imported
if __name__ == '__main__':
    main()