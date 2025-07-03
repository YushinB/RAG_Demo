# Simple RAG with Web Content

This project demonstrates a simple Retrieval-Augmented Generation (RAG) application using Streamlit, OpenAI, and web content. The app allows you to fetch text from a website, embed the content, save/load embeddings, and ask questions based on the embedded knowledge.

## Features

- **Fetch and embed website content:** Enter a URL, fetch its text, and create embeddings using OpenAI's embedding API.
- **Chunking:** Website text is split into manageable chunks for embedding.
- **Save and load embeddings:** Save embeddings and chunks to a `.pkl` file and reload them later.
- **Question answering:** Ask questions in a chat interface. The app retrieves the most relevant chunks and uses OpenAI's GPT model to answer your question using the context.

## Requirements

- Python 3.8+
- [OpenAI API key](https://platform.openai.com/)
- The following Python packages:
  - streamlit
  - python-dotenv
  - requests
  - beautifulsoup4
  - openai
  - numpy
  - scikit-learn

## Installation

1. **Clone the repository:**
    ```sh
    git clone <your-repo-url>
    cd rag-demo
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Set up environment variables:**
    - Create a `.env` file in the project root:
      ```
      OPENAI_API_KEY=your_openai_api_key_here
      ```

## Usage

1. **Run the Streamlit app:**
    ```sh
    streamlit run rag_app.py
    ```

2. **In the web UI:**
    - Enter a website URL and click "Embed Content" to fetch and embed the content.
    - Optionally, click "Embed Content and save to *.pkl file" to save embeddings for later use.
    - Click "Load the embedded *.pkl file" to reload saved embeddings.
    - Use the chat input to ask questions about the embedded content.

## File Structure

- `rag_app.py` — Main Streamlit application.
- `embeddings.pkl` — Saved embeddings and chunks (generated at runtime).
- `README.md` — This file.

## Notes

- The app uses OpenAI's `text-embedding-3-small` model for embeddings and `gpt-4.1-nano` for answering questions.
- Make sure your OpenAI API key has access to these models.
- For best results, use websites with mostly textual content.

## License

MIT