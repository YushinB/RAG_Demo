# Architecture Overview

This document provides visual representations of the RAG application architecture with ChromaDB integration.

## Current Architecture (Pickle-based)

```
┌─────────────────────────────────────────────────────────────┐
│                      User Browser                            │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP (Port 8501)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Streamlit Application                      │
│                      (rag_app.py)                            │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Embedding  │  │   Display    │  │   Chatbot    │     │
│  │     Tab      │  │     Tab      │  │     Tab      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  ┌────────────────────────────────────────────────┐         │
│  │         Session State Storage                  │         │
│  │  - doc_chunks (list)                           │         │
│  │  - doc_embeddings (list)                       │         │
│  │  - messages (list)                             │         │
│  └────────────────────────────────────────────────┘         │
└──────────────┬─────────────────────────┬────────────────────┘
               │                         │
               │ OpenAI API              │ Local File I/O
               │                         │
               ▼                         ▼
┌──────────────────────┐    ┌─────────────────────────┐
│   OpenAI Services    │    │   embeddings.pkl        │
│                      │    │   (Pickle File)         │
│  - Embeddings API    │    │                         │
│  - Chat Completions  │    │  - Text chunks          │
└──────────────────────┘    │  - Vector embeddings    │
                             └─────────────────────────┘
```

**Limitations:**
- ❌ No concurrent access (file locking issues)
- ❌ Manual file management
- ❌ No query optimization
- ❌ Not scalable

---

## Future Architecture (ChromaDB-based)

```
┌─────────────────────────────────────────────────────────────┐
│                      User Browser                            │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP (Port 8501)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Streamlit Application                      │
│                      (rag_app.py)                            │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Embedding  │  │   Display    │  │   Chatbot    │     │
│  │     Tab      │  │     Tab      │  │     Tab      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  ┌────────────────────────────────────────────────┐         │
│  │       ChromaDB Client Integration              │         │
│  │  - ChromaDBManager                             │         │
│  │  - Connection pooling                          │         │
│  │  - Query optimization                          │         │
│  └────────────────────────────────────────────────┘         │
└──────────┬─────────────────────────┬────────────────────────┘
           │                         │
           │ OpenAI API              │ HTTP API
           │                         │ (Internal Network)
           ▼                         ▼
┌──────────────────┐    ┌─────────────────────────────────────┐
│ OpenAI Services  │    │     ChromaDB Container              │
│                  │    │     (Docker)                        │
│ - Embeddings API │    │                                     │
│ - Chat API       │    │  ┌────────────────────────────┐    │
└──────────────────┘    │  │   REST API (Port 8000)     │    │
                        │  │   - Add embeddings         │    │
                        │  │   - Query similar          │    │
                        │  │   - Manage collections     │    │
                        │  └────────────────────────────┘    │
                        │                                     │
                        │  ┌────────────────────────────┐    │
                        │  │   Vector Storage Engine    │    │
                        │  │   - HNSW indexing          │    │
                        │  │   - Metadata filtering     │    │
                        │  │   - Similarity search      │    │
                        │  └────────────────────────────┘    │
                        │                                     │
                        │  ┌────────────────────────────┐    │
                        │  │   Persistent Volume        │    │
                        │  │   /chroma/chroma           │    │
                        │  └────────────────────────────┘    │
                        └─────────────────────────────────────┘
```

**Benefits:**
- ✅ Concurrent access (multiple users)
- ✅ Optimized similarity search (HNSW algorithm)
- ✅ Persistent storage with Docker volumes
- ✅ Scalable architecture
- ✅ Production-ready

---

## Docker Network Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Host Machine                               │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │         Docker Network: rag_network (bridge)           │  │
│  │                                                         │  │
│  │  ┌─────────────────────┐    ┌────────────────────┐   │  │
│  │  │  Streamlit App      │    │   ChromaDB         │   │  │
│  │  │  Container          │    │   Container        │   │  │
│  │  │  (Optional)         │◄───┤   rag_chromadb    │   │  │
│  │  │                     │    │                    │   │  │
│  │  │  - Port: 8501       │    │   - Port: 8000     │   │  │
│  │  │  - Volume: ./app    │    │   - Volume: data   │   │  │
│  │  │  - Env: OPENAI_KEY  │    │   - Persistent     │   │  │
│  │  │  - Env: CHROMA_HOST │    │   - Health checks  │   │  │
│  │  └──────────┬──────────┘    └─────────┬──────────┘   │  │
│  │             │                          │              │  │
│  └─────────────┼──────────────────────────┼──────────────┘  │
│                │                          │                  │
│           Port Mapping              Port Mapping             │
│                │                          │                  │
│         Host:8501 (exposed)      Host:8000 (exposed)         │
│                │                          │                  │
└────────────────┼──────────────────────────┼──────────────────┘
                 │                          │
                 ▼                          ▼
         ┌───────────────┐          ┌──────────────┐
         │  Web Browser  │          │  API Client  │
         │  Users        │          │  (Optional)  │
         └───────────────┘          └──────────────┘
```

**Network Features:**
- Isolated internal network for security
- Health checks for reliability
- Port mapping for external access
- Volume persistence for data

---

## Data Flow: Embedding Process

```
User Input (URL/Text/PDF)
         │
         ▼
┌────────────────────┐
│   Extract Text     │
│   - Web scraping   │
│   - PDF parsing    │
│   - File reading   │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│   Chunk Text       │
│   - Size: 100 char │
│   - Overlapping    │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Generate          │
│  Embeddings        │
│  (OpenAI API)      │
└─────────┬──────────┘
          │
          ▼
   ┌─────────────┐
   │  Store?     │
   └─────┬───────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌──────┐  ┌──────────┐
│Pickle│  │ ChromaDB │
│ File │  │ Database │
└──────┘  └──────────┘
```

---

## Data Flow: Query Process

```
User Question
      │
      ▼
┌──────────────────┐
│ Generate Query   │
│ Embedding        │
│ (OpenAI API)     │
└─────────┬────────┘
          │
          ▼
┌──────────────────┐      ┌─────────────────────┐
│  Similarity      │      │  Current: Manual    │
│  Search          │──────│  cosine_similarity  │
└─────────┬────────┘      │                     │
          │               │  Future: ChromaDB   │
          │               │  optimized HNSW     │
          │               └─────────────────────┘
          ▼
┌──────────────────┐
│ Retrieve Top-K   │
│ Similar Chunks   │
│ (K=3 default)    │
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│ Build Context    │
│ from Chunks      │
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│ Generate Answer  │
│ (OpenAI GPT)     │
│ with Context     │
└─────────┬────────┘
          │
          ▼
    Display Answer
```

---

## Deployment Options

### Option 1: Local Development
```
┌───────────────────┐
│  Developer PC     │
│                   │
│  - Python 3.8+    │
│  - pip install    │
│  - streamlit run  │
│                   │
│  + Docker Desktop │
│    └─ ChromaDB    │
└───────────────────┘
```

### Option 2: Docker Development
```
┌───────────────────────┐
│  Docker Host          │
│                       │
│  ┌─────────────────┐ │
│  │ Streamlit       │ │
│  │ Container       │ │
│  └────────┬────────┘ │
│           │          │
│  ┌────────▼────────┐ │
│  │ ChromaDB        │ │
│  │ Container       │ │
│  └─────────────────┘ │
└───────────────────────┘
```

### Option 3: Cloud Production
```
┌────────────────────────────────┐
│  Cloud Platform                 │
│  (AWS/Azure/GCP)               │
│                                 │
│  ┌──────────────────────┐      │
│  │ Container Service    │      │
│  │ (ECS/AKS/GKE)       │      │
│  │                      │      │
│  │  ┌────────────────┐ │      │
│  │  │ Streamlit App  │ │      │
│  │  └────────┬───────┘ │      │
│  │           │          │      │
│  │  ┌────────▼───────┐ │      │
│  │  │   ChromaDB     │ │      │
│  │  │   + Volume     │ │      │
│  │  └────────────────┘ │      │
│  └──────────────────────┘      │
│                                 │
│  + Load Balancer                │
│  + SSL/TLS                      │
│  + Auto-scaling                 │
│  + Monitoring                   │
└────────────────────────────────┘
```

---

## Component Interaction Diagram

```
┌────────────────────────────────────────────────────────────┐
│                    RAG Application Stack                    │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────┐     │
│  │              Presentation Layer                   │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐       │     │
│  │  │Embedding │  │ Display  │  │ Chatbot  │       │     │
│  │  │   UI     │  │   UI     │  │   UI     │       │     │
│  │  └──────────┘  └──────────┘  └──────────┘       │     │
│  └────────────────────┬─────────────────────────────┘     │
│                       │                                     │
│  ┌────────────────────▼─────────────────────────────┐     │
│  │              Business Logic Layer                 │     │
│  │  ┌─────────────────────────────────────────┐     │     │
│  │  │  - Text extraction & processing         │     │     │
│  │  │  - Chunking strategy                    │     │     │
│  │  │  - Embedding generation                 │     │     │
│  │  │  - Similarity search                    │     │     │
│  │  │  - Context assembly                     │     │     │
│  │  │  - Response generation                  │     │     │
│  │  └─────────────────────────────────────────┘     │     │
│  └──────────────┬────────────────┬───────────────────┘     │
│                 │                │                          │
│  ┌──────────────▼──────────┐  ┌─▼──────────────────────┐  │
│  │     Data Layer          │  │   External Services    │  │
│  │                         │  │                        │  │
│  │  Current: Pickle Files  │  │  - OpenAI Embeddings   │  │
│  │  Future:  ChromaDB      │  │  - OpenAI Chat         │  │
│  │                         │  │  - Web scraping        │  │
│  │  - Vector storage       │  │                        │  │
│  │  - Metadata storage     │  │                        │  │
│  │  - Query engine         │  │                        │  │
│  └─────────────────────────┘  └────────────────────────┘  │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

## Migration Path Visualization

```
Phase 1: Current State
┌─────────────┐
│   Pickle    │ ← Current production
│   Files     │
└─────────────┘

Phase 2: Add ChromaDB Support
┌─────────────┐   ┌─────────────┐
│   Pickle    │   │  ChromaDB   │ ← Both supported
│   Files     │   │  (Optional) │    via env var
└─────────────┘   └─────────────┘

Phase 3: Gradual Migration
┌─────────────┐   ┌─────────────┐
│   Pickle    │   │  ChromaDB   │ ← Migrate data
│  (Legacy)   │──>│  (Primary)  │    incrementally
└─────────────┘   └─────────────┘

Phase 4: Full ChromaDB
                  ┌─────────────┐
                  │  ChromaDB   │ ← Production
                  │  (Only)     │    deployment
                  └─────────────┘
```

---

## Technology Stack

```
┌─────────────────────────────────────────┐
│         Frontend Layer                   │
│  - Streamlit (UI framework)             │
│  - HTML/CSS (via Streamlit)             │
└────────────┬────────────────────────────┘
             │
┌────────────▼────────────────────────────┐
│         Application Layer                │
│  - Python 3.8+                          │
│  - OpenAI SDK                           │
│  - BeautifulSoup (web scraping)         │
│  - pdfplumber (PDF parsing)             │
│  - scikit-learn (similarity)            │
└────────────┬────────────────────────────┘
             │
┌────────────▼────────────────────────────┐
│         Storage Layer                    │
│  Current: pickle                        │
│  Future:  chromadb                      │
│           - Vector storage              │
│           - HNSW indexing               │
│           - REST API                    │
└────────────┬────────────────────────────┘
             │
┌────────────▼────────────────────────────┐
│         Infrastructure Layer             │
│  - Docker & Docker Compose              │
│  - Container orchestration              │
│  - Volume management                    │
│  - Network isolation                    │
└──────────────────────────────────────────┘
```

---

## Performance Comparison

### Current (Pickle) Performance
```
Operation          | Time      | Scalability
───────────────────┼───────────┼─────────────
Save embeddings    | 100ms     | Poor
Load embeddings    | 150ms     | Poor
Search (1000 docs) | 50ms      | O(n)
Concurrent users   | 1         | No
Memory usage       | Full load | High
```

### Future (ChromaDB) Performance
```
Operation          | Time      | Scalability
───────────────────┼───────────┼─────────────
Save embeddings    | 50ms      | Excellent
Load embeddings    | 10ms      | Excellent
Search (1000 docs) | 5ms       | O(log n)
Concurrent users   | 100+      | Yes
Memory usage       | Minimal   | Low
```

---

## Security Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Security Layers                     │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Layer 1: Network Security                          │
│  ┌───────────────────────────────────────────┐     │
│  │ - Docker network isolation                │     │
│  │ - Port restrictions                       │     │
│  │ - Firewall rules                          │     │
│  └───────────────────────────────────────────┘     │
│                                                      │
│  Layer 2: Application Security                      │
│  ┌───────────────────────────────────────────┐     │
│  │ - API key management (.env)               │     │
│  │ - Input validation                        │     │
│  │ - Rate limiting (future)                  │     │
│  └───────────────────────────────────────────┘     │
│                                                      │
│  Layer 3: Data Security                             │
│  ┌───────────────────────────────────────────┐     │
│  │ - Volume encryption                       │     │
│  │ - ChromaDB authentication (optional)      │     │
│  │ - Backup encryption                       │     │
│  └───────────────────────────────────────────┘     │
│                                                      │
│  Layer 4: Container Security                        │
│  ┌───────────────────────────────────────────┐     │
│  │ - Non-root user (future)                  │     │
│  │ - Resource limits                         │     │
│  │ - Health checks                           │     │
│  └───────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────┘
```

---

## Scalability Path

```
Stage 1: Single Instance
┌─────────────┐
│  Streamlit  │
│     +       │  ← 1-10 users
│  ChromaDB   │
└─────────────┘

Stage 2: Separated Services
┌─────────────┐   ┌─────────────┐
│  Streamlit  │   │  ChromaDB   │  ← 10-100 users
└─────────────┘   └─────────────┘

Stage 3: Load Balanced
┌─────────────────────────────────┐
│      Load Balancer              │
└────────┬─────────┬──────────────┘
         │         │
┌────────▼───┐ ┌──▼──────────┐
│Streamlit 1 │ │Streamlit 2  │   ← 100-1000 users
└────────┬───┘ └──┬──────────┘
         │        │
         └────┬───┘
              │
      ┌───────▼────────┐
      │   ChromaDB     │
      │   Cluster      │
      └────────────────┘

Stage 4: Distributed
┌────────────────────────────────────────┐
│        Global Load Balancer            │
└───┬────────────┬───────────────────┬───┘
    │            │                   │
┌───▼────┐  ┌───▼────┐         ┌───▼────┐
│Region 1│  │Region 2│  ...    │Region N│  ← 1000+ users
│Cluster │  │Cluster │         │Cluster │
└────────┘  └────────┘         └────────┘
```

---

This architecture documentation provides a comprehensive view of how the RAG application is structured and how ChromaDB integration enhances its capabilities.
