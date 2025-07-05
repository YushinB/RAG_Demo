# rag_app.py
# --- Description ---
# This is a simple RAG (Retrieval-Augmented Generation) application that allows users to embed website content and ask questions based on that content.
# It uses the OpenAI API for embeddings and a simple chat interface for interaction.
# --- Imports ---
# These are the libraries and modules that are imported for use in the application.
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
import time
import pdfplumber  # Library for PDF text extraction
import io

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

def extract_text_from_pdf(uploaded_file):
    """Extracts text from a PDF file using pdfplumber.
    Input:
        uploaded_file (UploadedFile): The PDF file uploaded by the user.
    Output:
        str: The extracted text from the PDF file.
    Meaning:
    This function takes a PDF file uploaded by the user and extracts its text content using the pdfplumber library.
    It reads each page of the PDF and concatenates the extracted text into a single string, which can then be used for further processing or embedding.
    """
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text
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

    Embedding_Tab, Display_Tab, RAG_Tab = st.tabs(["Embedding", "Embedded Results", "Retrieval Chatbot"])
# --- Section 1: Input URL & Embed ---
    with Embedding_Tab:

        # Initialize session state for document chunks and embeddings
        # This is the client that will be used to interact with the OpenAI API for embedding
        client = OpenAI(
        # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        # Initialize session state for document chunks and embeddings
        initialize_session_state()

         # --- Section 1: Input URL & Embed ---
        st.header("1. Embed Content from Various Sources")
        source_type = st.radio("Select content source:", ["Web URL", "Text", "PDF","*.pkl file" ])
        text_input = ""
        url = ""
        if source_type == "Web URL":
            urls = st.text_area(
                "Enter one or more website URLs (one per line):",
                placeholder="https://example.com\nhttps://another.com"
            )
            if urls:
                url_list = [u.strip() for u in urls.splitlines() if u.strip()]
                for url in url_list:
                    text_input += fetch_text_from_url(url).replace("\n", "")
        elif source_type == "Text":
            uploaded_text_file = st.file_uploader("Upload a text file", type=["txt"])
            if uploaded_text_file is not None:
                text_input = uploaded_text_file.read().decode("utf-8")
            else:
                text_input = ""
            if text_input:
                text_input = text_input.replace("\n", "")
        elif source_type == "PDF":
            uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
            if uploaded_file:
                text_input = extract_text_from_pdf(uploaded_file)
        elif source_type == "*.pkl file":
            uploaded_pkl_file = st.file_uploader("Upload an embedded *.pkl file", type=["pkl"])
            if uploaded_pkl_file is not None:
                try:
                    data = pickle.load(uploaded_pkl_file)
                    st.session_state.doc_chunks = data["chunks"]
                    st.session_state.doc_embeddings = data["embeddings"]
                    st.success("Loaded embeddings from uploaded file.")
                except Exception as e:
                    st.error(f"Failed to load embeddings: {e}")
        
        if source_type != "*.pkl file":
            embed_button = st.button("Embed Content")
            # Initialize OpenAI client
            # This is the client that will be used to interact with the OpenAI API for embedding
            client = OpenAI(
            # This is the default and can be omitted
                api_key=os.environ.get("OPENAI_API_KEY"),
            )

            # Initialize session state for document chunks and embeddings
            initialize_session_state()
            
            # --- Section 2: Display Embedded Content ---
            if embed_button and text_input:
                st.write(f"Fetched text from {url} with length: {len(text_input)} characters.")
                if len(text_input) < CHUNK_SIZE_SAVE:
                    st.error(f"Text is too short to chunk. Minimum length is {CHUNK_SIZE_SAVE} characters.")
                    return
                chunks = chunk_text(text_input, CHUNK_SIZE_EMBED)
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
            
            if st.session_state.get("doc_chunks") and st.session_state.get("doc_embeddings"):
                save_to_pkl_button = st.button("Save current embeddings to *.pkl file")
                if save_to_pkl_button:
                    custom_filename = st.text_input(
                            "Enter a custom filename for the embeddings (with .pkl extension):",
                            value=EMBEDDINGS_FILE,
                            key="custom_pkl_filename"
                        )
                    pkl_buffer = io.BytesIO()
                    pickle.dump(
                        {
                            "chunks": st.session_state.doc_chunks,
                            "embeddings": st.session_state.doc_embeddings
                        },
                        pkl_buffer
                    )
                    pkl_buffer.seek(0)
                    st.download_button(
                        label="Download Embedded Data",
                        data=pkl_buffer,
                        file_name=custom_filename if custom_filename else EMBEDDINGS_FILE,
                        mime="application/octet-stream"
                    )
                    st.success(f"✅ Embeddings ready for download as {custom_filename if custom_filename else EMBEDDINGS_FILE}")
                # Display the number of chunks and embeddings
                st.write(f"Number of chunks: {len(st.session_state.doc_chunks)}")   

# --- Section 2: Display Embedded Content ---
    with Display_Tab:
        st.header("Embedded Results")
        if st.session_state.get("doc_chunks") and st.session_state.get("doc_embeddings"):
            st.write(f"**Number of Chunks:** {len(st.session_state.doc_chunks)}")
            st.write(f"**Number of Embeddings:** {len(st.session_state.doc_embeddings)}")
            st.write("**Preview of Embedded Chunks:**")
            for i, chunk in enumerate(st.session_state.doc_chunks[:5]):
                st.markdown(f"**Chunk {i+1}:** {chunk[:300]}{'...' if len(chunk) > 300 else ''}")
        else:
            st.info("No embedded content available. Please embed content first.")
# --- Section 2: RAG Chatbot ---
    with RAG_Tab:
        # --- Section 3: RAG Chatbot ---
        st.header("2. Ask Questions")
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # React to user input
        if prompt := st.chat_input("Say something"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Simulate assistant response (replace with actual model call)
            if not st.session_state.doc_embeddings:
                st.warning("Please embed a website first.")
                return
            # Initialize OpenAI client
            # Display assistant response in chat message container
            response = "This is a simulated response. Replace with actual model call."
            with st.chat_message("assistant"):
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
                    response = f"Answer: {completion.choices[0].message.content}"
                    st.markdown(response)
                else:
                    st.warning("Please embed a website first.")
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        # Clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.success("Chat history cleared.")

# --- Main Function ---
if __name__ == '__main__':
    main()