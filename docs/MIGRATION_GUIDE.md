# Migration Guide: From Pickle Files to ChromaDB

This guide explains how to migrate from pickle file-based storage to ChromaDB vector database.

## Why Migrate?

| Feature | Pickle Files | ChromaDB |
|---------|-------------|----------|
| **Concurrent Access** | ❌ Single user only | ✅ Multiple users simultaneously |
| **Data Persistence** | ⚠️ File-based, manual backup | ✅ Database with ACID properties |
| **Scalability** | ❌ Limited to memory | ✅ Handles millions of vectors |
| **Query Performance** | ⚠️ Linear search | ✅ Optimized indexing (HNSW) |
| **Metadata Support** | ❌ Manual implementation | ✅ Built-in filtering |
| **Production Ready** | ❌ Development only | ✅ Production-grade |
| **API Access** | ❌ File system only | ✅ REST API available |
| **Docker Support** | N/A | ✅ Native container support |

## Current Implementation (Pickle)

The current application uses pickle files to store embeddings:

```python
# Save embeddings
with open("embeddings.pkl", "wb") as f:
    pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)

# Load embeddings
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    chunks = data["chunks"]
    embeddings = data["embeddings"]
```

**Limitations:**
- No concurrent access (file locking issues)
- No version control
- No incremental updates
- Manual file management
- Not suitable for web deployment

## Future Implementation (ChromaDB)

ChromaDB provides a robust alternative:

```python
import chromadb
from chromadb.config import Settings

# Initialize client
client = chromadb.HttpClient(host='localhost', port=8000)

# Create/get collection
collection = client.get_or_create_collection(
    name="rag_documents",
    metadata={"description": "RAG embedded documents"}
)

# Add embeddings
collection.add(
    embeddings=embeddings,
    documents=chunks,
    ids=[f"doc_{i}" for i in range(len(chunks))],
    metadatas=[{"source": url, "chunk_id": i} for i in range(len(chunks))]
)

# Query similar documents
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)
```

**Benefits:**
- Persistent storage with database guarantees
- Concurrent access from multiple clients
- Metadata filtering
- Automatic similarity search optimization
- Easy to scale and deploy

## Migration Steps

### Phase 1: Setup ChromaDB

1. **Start ChromaDB container:**
   ```bash
   docker-compose up -d chromadb
   ```

2. **Install Python client:**
   ```bash
   pip install chromadb
   ```

3. **Verify connection:**
   ```python
   import chromadb
   client = chromadb.HttpClient(host='localhost', port=8000)
   print(client.heartbeat())  # Should return timestamp
   ```

### Phase 2: Create Migration Script

Create `migrate_pickle_to_chromadb.py`:

```python
import chromadb
import pickle
import sys

def migrate_embeddings(pkl_file, collection_name="rag_documents"):
    """Migrate pickle file embeddings to ChromaDB"""
    
    # Load pickle data
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    chunks = data['chunks']
    embeddings = data['embeddings']
    
    # Connect to ChromaDB
    client = chromadb.HttpClient(host='localhost', port=8000)
    
    # Create collection
    collection = client.get_or_create_collection(name=collection_name)
    
    # Add data in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        batch_ids = [f"doc_{j}" for j in range(i, min(i+batch_size, len(chunks)))]
        
        collection.add(
            embeddings=batch_embeddings,
            documents=batch_chunks,
            ids=batch_ids
        )
        print(f"Migrated {min(i+batch_size, len(chunks))}/{len(chunks)} documents")
    
    print(f"Migration complete! {len(chunks)} documents in collection '{collection_name}'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python migrate_pickle_to_chromadb.py <pickle_file.pkl>")
        sys.exit(1)
    
    migrate_embeddings(sys.argv[1])
```

### Phase 3: Run Migration

```bash
python migrate_pickle_to_chromadb.py embeddings.pkl
```

### Phase 4: Update Application Code

See `docs/chromadb_integration_example.py` for complete code changes needed in `rag_app.py`.

Key changes:
1. Add ChromaDB client initialization
2. Replace pickle save/load with ChromaDB operations
3. Add collection management
4. Update similarity search to use ChromaDB queries

### Phase 5: Test Both Methods

During transition, support both storage methods:

```python
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "pickle")  # or "chromadb"

if STORAGE_BACKEND == "chromadb":
    # Use ChromaDB
    collection.add(...)
else:
    # Use pickle (legacy)
    save_embeddings(chunks, embeddings)
```

## Migration Checklist

- [ ] Start ChromaDB container (`docker-compose up -d chromadb`)
- [ ] Install ChromaDB Python client (`pip install chromadb`)
- [ ] Test ChromaDB connection
- [ ] Create migration script
- [ ] Backup existing pickle files
- [ ] Run migration for test data
- [ ] Verify data in ChromaDB
- [ ] Update application code
- [ ] Test application with ChromaDB backend
- [ ] Run side-by-side comparison (pickle vs ChromaDB)
- [ ] Deploy to production
- [ ] Monitor performance
- [ ] Archive old pickle files

## Rollback Plan

If issues occur:

1. **Keep pickle files as backup**
2. **Use environment variable to switch backends:**
   ```bash
   export STORAGE_BACKEND=pickle
   streamlit run rag_app.py
   ```
3. **ChromaDB data is persistent** in Docker volume
4. **Can export ChromaDB back to pickle if needed**

## Performance Comparison

Based on typical RAG workloads:

| Operation | Pickle (1000 docs) | ChromaDB (1000 docs) |
|-----------|-------------------|---------------------|
| **Save embeddings** | ~100ms | ~50ms |
| **Load embeddings** | ~150ms | ~10ms |
| **Search top-3** | ~50ms | ~5ms |
| **Concurrent users** | 1 | 100+ |
| **Memory usage** | Full dataset | Minimal |

*Results may vary based on hardware and dataset size*

## Best Practices

1. **Use collections for different documents:** Separate collections for different data sources
2. **Add meaningful metadata:** Include source URLs, timestamps, chunk indices
3. **Implement error handling:** Network issues, connection failures
4. **Regular backups:** Use ChromaDB backup tools
5. **Monitor performance:** Track query times and resource usage
6. **Version your data:** Use metadata to track data versions

## Troubleshooting

### Migration fails with "Connection refused"

**Solution:** Ensure ChromaDB is running:
```bash
docker-compose ps chromadb
curl http://localhost:8000/api/v1/heartbeat
```

### "Collection already exists" error

**Solution:** Either delete existing collection or use different name:
```python
client.delete_collection("rag_documents")
# OR
collection_name = f"rag_documents_{timestamp}"
```

### Embeddings don't match pickle format

**Solution:** Verify embedding dimensions:
```python
print(f"Embedding dimension: {len(embeddings[0])}")
# Should match ChromaDB collection configuration
```

### Performance slower than expected

**Solution:** 
- Increase batch size for bulk operations
- Use ChromaDB's built-in indexing
- Check network latency to ChromaDB
- Consider using persistent client instead of HttpClient for local deployments

## Advanced: Hybrid Approach

Support both backends simultaneously:

```python
class EmbeddingStore:
    def __init__(self, backend="chromadb"):
        self.backend = backend
        if backend == "chromadb":
            self.client = chromadb.HttpClient(host='localhost', port=8000)
            self.collection = self.client.get_or_create_collection("rag_docs")
    
    def add(self, chunks, embeddings):
        if self.backend == "chromadb":
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                ids=[f"doc_{i}" for i in range(len(chunks))]
            )
        else:
            save_embeddings(chunks, embeddings)
    
    def query(self, query_embedding, k=3):
        if self.backend == "chromadb":
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            return results['documents'][0], results['distances'][0]
        else:
            # Use existing pickle-based search
            return get_top_k_similar_docs(query_embedding, embeddings, k)
```

## Next Steps

1. **Test migration with sample data**
2. **Deploy ChromaDB to staging environment**
3. **Update monitoring and alerting**
4. **Plan production migration**
5. **Document any customizations**

## Support

For issues during migration:
- Check [Docker Setup Guide](DOCKER_SETUP.md) for ChromaDB configuration
- Review [Vector Database Investigation](vector_database_investigation.md) for technical details
- Consult ChromaDB documentation: https://docs.trychroma.com/

---

**Note:** The current implementation continues to use pickle files. This migration guide is for future enhancement when transitioning to a production vector database.
