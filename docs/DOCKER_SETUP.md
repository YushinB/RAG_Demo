# Docker Setup Guide for RAG Application

This guide explains how to deploy the RAG application with ChromaDB vector database using Docker.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration Options](#configuration-options)
- [Architecture](#architecture)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Docker Engine 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- Docker Compose 2.0+ ([Install Docker Compose](https://docs.docker.com/compose/install/))
- OpenAI API Key ([Get API Key](https://platform.openai.com/))

## Quick Start

### Option 1: ChromaDB Only (Recommended for Development)

Run only the ChromaDB vector database and run the Streamlit app locally:

1. **Start ChromaDB:**
   ```bash
   docker-compose up -d chromadb
   ```

2. **Verify ChromaDB is running:**
   ```bash
   docker-compose ps
   curl http://localhost:8000/api/v1/heartbeat
   ```

3. **Set up environment variables:**
   ```bash
   cp sample.env .env
   # Edit .env and add your OPENAI_API_KEY
   ```

4. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Streamlit app:**
   ```bash
   streamlit run rag_app.py
   ```

6. **Access the application:**
   Open your browser to `http://localhost:8501`

### Option 2: Full Stack with Docker (Production-like)

Run both ChromaDB and the Streamlit app in containers:

1. **Update docker-compose.yml:**
   
   Uncomment the `streamlit_app` service section in `docker-compose.yml`

2. **Create .env file:**
   ```bash
   cp sample.env .env
   # Edit .env and add:
   # OPENAI_API_KEY=your_key_here
   # ANONYMIZED_TELEMETRY=FALSE  # Optional: disable telemetry
   ```

3. **Start all services:**
   ```bash
   docker-compose up -d
   ```

4. **Check service status:**
   ```bash
   docker-compose ps
   docker-compose logs -f
   ```

5. **Access the application:**
   - Streamlit App: `http://localhost:8501`
   - ChromaDB API: `http://localhost:8000`

## Configuration Options

### Environment Variables

Create a `.env` file in the project root:

```env
# Required
OPENAI_API_KEY=sk-your-openai-api-key

# Optional
ANONYMIZED_TELEMETRY=FALSE
CHROMA_HOST=chromadb
CHROMA_PORT=8000
```

### Docker Compose Configuration

#### Ports

- **8000**: ChromaDB API (HTTP)
- **8501**: Streamlit application

To change ports, edit `docker-compose.yml`:

```yaml
ports:
  - "9000:8000"  # Map host port 9000 to container port 8000
```

#### Volumes

ChromaDB data is persisted in a named volume:

```yaml
volumes:
  chromadb_data:
    driver: local
```

To use a host directory instead:

```yaml
volumes:
  - ./chromadb_data:/chroma/chroma
```

#### Memory Limits

Add resource limits for production:

```yaml
services:
  chromadb:
    # ... other config ...
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          memory: 512M
```

## Architecture

### Docker Network Architecture

```
┌─────────────────────────────────────────────┐
│              Host Machine                    │
│                                              │
│  ┌────────────────────────────────────┐    │
│  │     Docker Network (rag_network)   │    │
│  │                                     │    │
│  │  ┌──────────────┐  ┌────────────┐ │    │
│  │  │  Streamlit   │  │  ChromaDB  │ │    │
│  │  │     App      │◄─┤  Vector DB │ │    │
│  │  │ Port: 8501   │  │ Port: 8000 │ │    │
│  │  └──────┬───────┘  └─────┬──────┘ │    │
│  │         │                 │        │    │
│  └─────────┼─────────────────┼────────┘    │
│            │                 │              │
│         Host:8501        Host:8000          │
│            │                 │              │
└────────────┼─────────────────┼──────────────┘
             │                 │
          Browser          API Access
```

### Data Flow

1. User interacts with Streamlit UI (port 8501)
2. App processes content and generates embeddings (OpenAI API)
3. Embeddings stored in ChromaDB (port 8000)
4. Queries retrieve relevant vectors from ChromaDB
5. GPT generates answers based on retrieved context

## Usage

### Starting Services

```bash
# Start all services in background
docker-compose up -d

# Start with logs visible
docker-compose up

# Start only ChromaDB
docker-compose up -d chromadb
```

### Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (deletes all data)
docker-compose down -v
```

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f chromadb
docker-compose logs -f streamlit_app
```

### Accessing ChromaDB Directly

ChromaDB provides a REST API accessible at `http://localhost:8000`:

```bash
# Health check
curl http://localhost:8000/api/v1/heartbeat

# List collections
curl http://localhost:8000/api/v1/collections
```

### Backing Up Data

```bash
# Create backup of ChromaDB volume
docker run --rm -v rag_demo_chromadb_data:/data -v $(pwd):/backup \
  ubuntu tar czf /backup/chromadb_backup_$(date +%Y%m%d).tar.gz -C /data .
```

### Restoring Data

```bash
# Restore from backup
docker run --rm -v rag_demo_chromadb_data:/data -v $(pwd):/backup \
  ubuntu tar xzf /backup/chromadb_backup_YYYYMMDD.tar.gz -C /data
```

## Troubleshooting

### ChromaDB Container Won't Start

**Check logs:**
```bash
docker-compose logs chromadb
```

**Common issues:**
- Port 8000 already in use: Change port in `docker-compose.yml`
- Volume permission issues: Check volume mount permissions
- Insufficient disk space: Clean up Docker system

**Fix disk space:**
```bash
docker system prune -a
```

### Can't Connect to ChromaDB

**Verify container is running:**
```bash
docker-compose ps
```

**Check network connectivity:**
```bash
docker-compose exec streamlit_app ping chromadb
```

**Test ChromaDB directly:**
```bash
curl http://localhost:8000/api/v1/heartbeat
```

### Streamlit App Can't Access ChromaDB

**Ensure services are on same network:**
```bash
docker network inspect rag_demo_rag_network
```

**Check environment variables:**
```bash
docker-compose exec streamlit_app env | grep CHROMA
```

### Data Not Persisting

**Check volume exists:**
```bash
docker volume ls | grep chromadb
```

**Inspect volume:**
```bash
docker volume inspect rag_demo_chromadb_data
```

**Verify mount in container:**
```bash
docker-compose exec chromadb ls -la /chroma/chroma
```

### Performance Issues

**Check resource usage:**
```bash
docker stats
```

**Increase memory limits in docker-compose.yml**

**Optimize ChromaDB settings** (add to docker-compose.yml):
```yaml
environment:
  - CHROMA_SERVER_CORS_ALLOW_ORIGINS=["*"]
  - CHROMA_WORKERS=4
```

### API Key Issues

**Verify API key is set:**
```bash
docker-compose exec streamlit_app env | grep OPENAI
```

**Check .env file format:**
- No spaces around `=`
- No quotes around values (unless value contains spaces)
- File should be in same directory as docker-compose.yml

### Recreate Services

```bash
# Full reset (WARNING: deletes data)
docker-compose down -v
docker-compose up -d
```

## Security Best Practices

1. **Never commit `.env` file** - Added to `.gitignore`
2. **Use secrets management in production** - Consider Docker Secrets or HashiCorp Vault
3. **Enable authentication** - Configure ChromaDB authentication for production
4. **Use HTTPS** - Add reverse proxy (nginx/traefik) with SSL certificates
5. **Restrict network access** - Use firewall rules to limit container access
6. **Regular updates** - Keep images updated: `docker-compose pull`

## Production Deployment

For production deployment, consider:

1. **Use specific version tags** instead of `latest`
2. **Set up monitoring** (Prometheus, Grafana)
3. **Configure log rotation**
4. **Use external persistent storage** (NFS, EBS, etc.)
5. **Set up automated backups**
6. **Enable authentication on ChromaDB**
7. **Use container orchestration** (Kubernetes, Docker Swarm)
8. **Implement health checks and auto-restart policies**

Example production docker-compose.yml additions:

```yaml
services:
  chromadb:
    image: chromadb/chroma:0.4.22  # Pin version
    restart: always
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
```

## Additional Resources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Streamlit Docker Deployment](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)
- [Vector Database Investigation Report](./vector_database_investigation.md)

## Support

For issues or questions:
1. Check this troubleshooting guide
2. Review ChromaDB logs: `docker-compose logs chromadb`
3. Check application logs: `docker-compose logs streamlit_app`
4. Consult the [vector database investigation document](./vector_database_investigation.md)
