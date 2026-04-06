# Compound ML Setup Guide

## Python Requirements

Python 3.10 or later is required. Verify with:

```bash
python3 --version
```

## Core Packages (Required)

These packages are needed for basic data exploration and profiling:

```bash
pip install pandas scikit-learn matplotlib
```

Or with uv:

```bash
uv pip install pandas scikit-learn matplotlib
```

## Clustering and Anomaly Detection Packages

For unsupervised clustering and dimensionality reduction:

```bash
pip install umap-learn hdbscan
```

Note: `hdbscan` has C extensions and may require a C compiler. On macOS, install Xcode command line tools (`xcode-select --install`). On Linux, install `build-essential`.

## Embedding Providers

Embeddings improve the quality of clustering and anomaly detection on text data. Without embeddings, skills fall back to TF-IDF (lower quality but always available).

### Option A: OpenAI Embeddings (Recommended)

Set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY=sk-...
```

This uses the `text-embedding-3-small` model by default. Embedding costs are typically under $0.01 per 1000 rows of text data.

### Option B: Local Embeddings (Free, No API Key)

Install sentence-transformers for local embedding generation:

```bash
pip install sentence-transformers
```

This downloads a ~100MB model on first use. Slower than API embeddings but completely free and private.

### Option C: No Embeddings (TF-IDF Fallback)

If neither option is configured, skills that need text representations fall back to TF-IDF vectorization via scikit-learn. This works but produces lower-quality clusters and anomaly detection for text data. Numeric data does not need embeddings.

**Note:** RAG pipelines (`ml-rag`) require real embeddings (Option A or B). TF-IDF is not sufficient for retrieval-augmented generation.

## RAG Pipeline Packages

For building and querying document retrieval pipelines:

```bash
pip install chromadb rank-bm25
```

ChromaDB provides local vector storage (no external server needed). rank-bm25 adds keyword-based retrieval for hybrid search.

## Quick Verification

Run this to check your environment:

```bash
python3 -c "
import pandas; print(f'pandas {pandas.__version__}')
import sklearn; print(f'scikit-learn {sklearn.__version__}')
import matplotlib; print(f'matplotlib {matplotlib.__version__}')
try:
    import umap; print(f'umap-learn {umap.__version__}')
except ImportError: print('umap-learn: not installed (needed for clustering)')
try:
    import hdbscan; print(f'hdbscan {hdbscan.__version__}')
except ImportError: print('hdbscan: not installed (needed for clustering)')
try:
    import sentence_transformers; print('sentence-transformers: installed')
except ImportError: print('sentence-transformers: not installed')
try:
    import chromadb; print(f'chromadb {chromadb.__version__}')
except ImportError: print('chromadb: not installed (needed for RAG)')
import os
if os.environ.get('OPENAI_API_KEY'): print('OpenAI API key: configured')
else: print('OpenAI API key: not set')
"
```
