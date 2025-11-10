# Quick Start Guide

Get up and running with the RAG application in under 5 minutes!

## Prerequisites

- Docker and Docker Compose installed
- OpenAI API key
- Python 3.8+ (for local development)

## 🚀 Fastest Way to Start

### 1. Clone and Configure

```bash
git clone <your-repo-url>
cd RAG_Demo
cp sample.env .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-key-here
```

### 2. Start ChromaDB (Vector Database)

```bash
docker-compose up -d chromadb
```

Verify it's running:
```bash
curl http://localhost:8000/api/v1/heartbeat
# Should return: {}
```

### 3. Install Dependencies & Run

```bash
pip install -r requirements.txt
streamlit run rag_app.py
```

### 4. Access the Application

Open your browser to: http://localhost:8501

## 🎯 What You Can Do

1. **Embed Content**:
   - Enter a website URL
   - Upload a PDF or text file
   - Content is automatically chunked and embedded

2. **View Results**:
   - See how many chunks were created
   - Preview the embedded content

3. **Ask Questions**:
   - Chat with your embedded content
   - Get AI-powered answers based on the context

## 🐳 Full Docker Deployment

To run everything in Docker:

1. Uncomment the `streamlit_app` section in `docker-compose.yml`
2. Run: `docker-compose up -d`
3. Access app at http://localhost:8501

## 📚 Need More Help?

- **Docker Details**: See [docs/DOCKER_SETUP.md](docs/DOCKER_SETUP.md)
- **Vector DB Info**: See [docs/vector_database_investigation.md](docs/vector_database_investigation.md)
- **Full README**: See [README.md](README.md)

## 🛑 Stop Services

```bash
docker-compose down
```

## 🔧 Troubleshooting

**ChromaDB won't start?**
- Check if port 8000 is available: `lsof -i :8000`
- View logs: `docker-compose logs chromadb`

**Can't connect to OpenAI?**
- Verify your API key in `.env`
- Check your OpenAI account has credits

**Streamlit issues?**
- Clear Streamlit cache: `streamlit cache clear`
- Check Python version: `python --version` (needs 3.8+)

## 💡 Pro Tips

- **Save embeddings**: Use the "Save to .pkl" button to avoid re-embedding
- **Multiple URLs**: Enter one URL per line for batch processing
- **Token usage**: Embeddings cost tokens, save them after creating!
- **Production**: Use ChromaDB for persistent, scalable storage

---

Happy RAGging! 🎉
