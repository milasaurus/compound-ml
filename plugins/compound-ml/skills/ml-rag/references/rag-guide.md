# How RAG Works (Plain Language)

## What Is RAG?

RAG stands for Retrieval-Augmented Generation. It lets an AI answer questions about YOUR documents — not just its general training knowledge. Think of it as giving the AI a searchable filing cabinet of your specific content.

## Why Not Just Paste Documents into the Chat?

AI models have a limited "context window" — the amount of text they can consider at once. If you have hundreds of documents, they won't all fit. RAG solves this by finding only the relevant pieces and giving those to the AI.

## How It Works

### Step 1: Build the Index (One-Time Setup)

1. **Read documents** — All your text files, markdown, PDFs are read in
2. **Split into chunks** — Long documents are broken into smaller pieces (a few paragraphs each), because a whole document might be too broad and a single sentence too narrow
3. **Create searchable representations** — Each chunk gets converted into a numerical representation (an "embedding") that captures its meaning. Similar content gets similar representations
4. **Store in a searchable database** — The chunks and their representations are saved locally so they can be searched quickly

### Step 2: Ask Questions (Repeatable)

1. **Understand the question** — Your question is also converted into a numerical representation
2. **Find relevant chunks** — Two search methods work together:
   - **Meaning-based search** — Finds chunks with similar meaning to your question (even if they use different words)
   - **Keyword search** — Finds chunks with the same specific terms
3. **Combine results** — Chunks found by both methods rank highest
4. **Generate an answer** — The most relevant chunks are given to the AI, which writes an answer based specifically on that content
5. **Cite sources** — The answer includes which documents and sections the information came from

## What Makes a Good RAG Pipeline

- **Good chunking** — Splitting documents at natural boundaries (headings, paragraphs) preserves meaning better than splitting at arbitrary character counts
- **Quality embeddings** — OpenAI or sentence-transformers produce better results than simple keyword matching
- **Hybrid retrieval** — Using both meaning-based and keyword search catches more relevant content than either alone
- **Focused context** — Giving the AI fewer, more relevant chunks produces better answers than dumping everything in

## Limitations

- **Only knows what's indexed** — If the answer isn't in your documents, RAG can't find it
- **Chunking can split context** — Sometimes the answer spans a section boundary. The overlap between chunks helps but doesn't eliminate this
- **Embedding quality matters** — TF-IDF (the simplest approach) doesn't understand meaning, just word overlap. For serious use, configure OpenAI or sentence-transformers embeddings
- **Stale indexes** — If you update your documents, rebuild the index to include the changes

## When to Use RAG

- **Internal knowledge bases** — Make company docs, wikis, or guides searchable
- **Research collections** — Ask questions across a large set of papers or notes
- **Code documentation** — Search across README files, API docs, and guides
- **Personal notes** — Query your own note archive with natural language

## When NOT to Use RAG

- **Small document sets** — If everything fits in the AI's context window, just paste it in directly
- **Highly structured data** — For databases or spreadsheets, use `ml-explore` or `ml-cluster` instead
- **Real-time data** — RAG works on a static index. For live data, you need a different approach
