---
name: ml-rag
description: "Build and query a retrieval-augmented generation pipeline from documents. Use when the user wants to create a searchable knowledge base, ask questions about a document collection, or says 'build a RAG pipeline', 'index these docs', 'search my documents', or 'answer questions from these files'."
argument-hint: "build <directory> | query <question>"
---

# Retrieval-Augmented Generation Pipeline

Build a searchable index from a collection of documents, then answer questions grounded in those documents with source citations. No ML expertise required — chunking, embedding, indexing, and retrieval are all handled automatically.

**Important:** Unlike other ml skills, RAG requires sentence-transformers for embeddings. TF-IDF is not sufficient for retrieval quality. If sentence-transformers is not installed, report this clearly and point to `references/setup.md`.

## Modes

This skill has two modes, specified as the first argument:

- **`build <directory>`** — Index a collection of documents into a searchable vector store
- **`query <question>`** — Ask a question and get an answer with source citations from the index

If no mode is specified, check if an index exists in `.ml-rag-index/`:
- If yes, enter query mode
- If no, ask the user what they want to do

## Build Mode

### Phase 1: Environment Check

Verify required packages:

```bash
python3 -c "
import chromadb; print(f'chromadb {chromadb.__version__}')
try:
    import sentence_transformers; print('sentence-transformers: available')
except ImportError: print('sentence-transformers: NOT installed')
"
```

**Required:** chromadb. If not installed:
> ChromaDB is required for RAG pipelines. Install it with:
> ```
> pip install chromadb
> ```

**Required:** sentence-transformers. If not installed:
> RAG requires sentence-transformers for quality retrieval. Install it with:
> ```
> pip install sentence-transformers
> ```
> See `references/setup.md` for detailed setup instructions.

Do not proceed if either requirement is unmet. TF-IDF is not a viable fallback for RAG.

### Phase 2: Discover and Read Documents

Scan the specified directory for documents:

```python
import os, pathlib
SUPPORTED = {'.md', '.txt', '.text', '.pdf', '.html', '.rst'}
docs = []
for root, dirs, files in os.walk(directory):
    for f in files:
        if pathlib.Path(f).suffix.lower() in SUPPORTED:
            docs.append(os.path.join(root, f))
```

Report: "Found [N] documents in [directory]"

If no supported files are found, report the supported formats and stop.

For each document, read the full content. For PDF files, attempt extraction with `pymupdf` or `pdfplumber` if available; otherwise skip PDFs with a note.

### Phase 3: Chunk Documents

Split documents into chunks suitable for retrieval:

```python
def chunk_document(text, source_path, target_tokens=400, overlap_tokens=50):
    """
    Semantic chunking: split on headings and paragraph boundaries.
    Target 256-512 tokens per chunk (roughly 400 words).
    Keep overlap for context continuity.
    """
    # Split on markdown headings (##, ###) or double newlines
    # Each chunk gets metadata: source file, chunk index, heading context
    ...
```

Chunking strategy:
- Split on heading boundaries (markdown `##`, `###`) first
- If a section is still too long, split on paragraph boundaries (double newline)
- If a paragraph is still too long, split on sentence boundaries
- Target chunk size: 256-512 tokens (~300-600 words)
- Overlap: 50 tokens between adjacent chunks for context continuity
- Each chunk carries metadata: source file path, chunk index, nearest heading

Report: "Created [N] chunks from [M] documents (average [X] tokens per chunk)"

### Phase 4: Generate Embeddings and Index

Generate embeddings using sentence-transformers:

```python
import chromadb

client = chromadb.PersistentClient(path=".ml-rag-index")
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)
```

Generate embeddings and add to ChromaDB in batches of 100:

```python
# ChromaDB can handle embedding generation if configured with an embedding function
# Or generate embeddings externally and pass them in
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=[c['text'] for c in chunks],
    metadatas=[{'source': c['source'], 'index': c['index'], 'heading': c['heading']} for c in chunks],
    embeddings=embeddings  # pre-computed embedding vectors
)
```

Report: "Index built successfully at `.ml-rag-index/` with [N] chunks from [M] documents."

### Phase 5: Build BM25 Index

Build a keyword index alongside the vector index:

```python
from rank_bm25 import BM25Okapi
import json

tokenized = [chunk['text'].lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenized)

# Save BM25 data for query mode
with open('.ml-rag-index/bm25_corpus.json', 'w') as f:
    json.dump({'chunks': [c['text'] for c in chunks], 'metadata': [c['metadata'] for c in chunks]}, f)
```

If rank-bm25 is not installed, skip BM25 and use vector-only retrieval with a note:
> "For better retrieval quality, install rank-bm25: `pip install rank-bm25`"

## Query Mode

### Phase 1: Load Index

Check that `.ml-rag-index/` exists:

```python
import chromadb
client = chromadb.PersistentClient(path=".ml-rag-index")
collection = client.get_collection("documents")
print(f"Index loaded: {collection.count()} chunks")
```

If the index does not exist:
> "No index found. Build one first with: `compound-ml:ml-rag build <directory>`"

### Phase 2: Retrieve Relevant Chunks

**Vector retrieval** via ChromaDB:

```python
results = collection.query(
    query_texts=[question],
    n_results=10
)
```

**BM25 retrieval** (if index exists):

```python
import json
from rank_bm25 import BM25Okapi

with open('.ml-rag-index/bm25_corpus.json') as f:
    data = json.load(f)

tokenized = [doc.lower().split() for doc in data['chunks']]
bm25 = BM25Okapi(tokenized)
bm25_scores = bm25.get_scores(question.lower().split())
top_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
```

**Merge and deduplicate:**

Combine results from both retrievers. Deduplicate by chunk ID. If a chunk appears in both result sets, boost its rank. Keep the top 10 unique chunks.

### Phase 3: Assemble Context and Generate Answer

Assemble the retrieved chunks into context, keeping total context under 8,000 tokens:

```
Context from indexed documents:

[Source: path/to/file.md, Section: "Authentication"]
<chunk text>

[Source: path/to/other.md, Section: "Setup"]
<chunk text>
...
```

Generate an answer using the LLM with this prompt structure:

> Answer the following question using ONLY the provided context. If the context does not contain enough information to answer, say so clearly. Always cite your sources.
>
> Question: [user's question]
>
> Context:
> [assembled chunks with source metadata]

### Phase 4: Present Answer

Format the response as:

```
## Answer

[Direct answer to the question in clear prose]

### Sources

- [filename.md], "[Section heading]" — [brief quote or paraphrase showing relevance]
- [other-file.md], "[Section heading]" — [brief quote or paraphrase]

---
*Answer generated from [N] indexed documents. If this doesn't fully answer your question, try rephrasing or check if the relevant documents are in the index.*
```

If the retrieved context does not contain relevant information:
> "I couldn't find relevant information in the indexed documents for this question. The index contains [N] documents about [general topics]. Try rephrasing your question, or check if the relevant documents are in the indexed directory."

## Error Handling

- **ChromaDB not installed:** Report install instructions, do not proceed
- **No sentence-transformers:** Report clearly that RAG requires sentence-transformers, point to setup.md
- **Empty directory:** "No supported documents found in [path]. Supported formats: .md, .txt, .pdf, .html, .rst"
- **Index does not exist (query mode):** Prompt to build first
- **Embedding API failure during indexing:** Save progress to ChromaDB (partial index is usable), report error
- **All chunks score low relevance:** Report that the question may not be covered by the indexed documents

## Reference Files

- `references/rag-guide.md` — Plain-language explanation of how RAG works
