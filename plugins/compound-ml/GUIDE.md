# Compound ML — Step-by-Step Guide

Give me any data — docs, logs, user behavior, API responses — and I'll surface patterns, anomalies, and answers in plain language using foundation models. No labels, no training pipeline, no ML expertise required.

## Prerequisites

You need:
- **Python 3.10+**
- **Claude Code** with the compound-ml plugin installed (`claude /plugin install compound-ml`)

Set up a virtual environment and install the core packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas scikit-learn matplotlib
```

For the full experience (better clustering, anomaly detection, and RAG):

```bash
pip install umap-learn hdbscan sentence-transformers chromadb rank-bm25
```

Remember to activate the virtual environment (`source venv/bin/activate`) before running any skills.

Verify your environment:

```bash
python3 -c "
import pandas; print(f'pandas {pandas.__version__}')
import sklearn; print(f'scikit-learn {sklearn.__version__}')
try: import umap; print(f'umap-learn {umap.__version__}')
except ImportError: print('umap-learn: not installed')
try: import hdbscan; print('hdbscan: installed')
except ImportError: print('hdbscan: not installed')
try: import sentence_transformers; print('sentence-transformers: installed')
except ImportError: print('sentence-transformers: not installed')
try: import chromadb; print(f'chromadb {chromadb.__version__}')
except ImportError: print('chromadb: not installed')
"
```

For detailed setup options (including embedding providers and platform-specific notes), see `skills/ml-explore/references/setup.md`.

## Quick Start

Download the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) from Kaggle (~170k tracks with audio features like danceability, energy, valence, and tempo). Unzip it, then:

```
# 1. See what you have
/compound-ml:ml-explore spotify_tracks_dataset.csv

# 2. Get a full analysis — profiling, grouping, and anomaly detection in one pass
/compound-ml:ml-analyze spotify_tracks_dataset.csv "find natural music genres and unusual tracks"
```

That's it. The plugin handles data loading, algorithm selection, parameter tuning, and result interpretation. You get a plain-language report.

If you want more control, read on — each skill below handles one part of the pipeline independently.

## The Workflow

Here's how the skills fit together:

```
                        ┌─────────────────┐
            Your Data ──│   ml-explore     │── Understand your data
                        └────────┬────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼             ▼
            ┌──────────┐ ┌─────────────┐ ┌─────────┐
            │ml-cluster│ │ml-anomalies │ │ ml-rag  │
            └──────────┘ └─────────────┘ └─────────┘
             Find groups  Flag outliers   Q&A over docs
                    │            │             │
                    └────────────┼─────────────┘
                                 ▼
                        ┌─────────────────┐
                        │   ml-analyze    │── Or do it all at once
                        └─────────────────┘
```

**Start with `ml-explore`** to understand your data, then pick a specific skill based on what you want to learn. Or skip straight to `ml-analyze` to run the full pipeline automatically.

<img width="1708" height="1022" alt="image" src="https://github.com/user-attachments/assets/c19556ac-060d-433d-8100-b552c2aff2e1" />

---

## Skills

### 1. Explore — Profile Your Data

**What it does:** Reads your dataset and produces a plain-language summary of its shape, columns, distributions, data quality issues, and interesting patterns.

**When to use it:** You just got a dataset and want to understand what's in it before deciding what to do next.

**How to run it:**

```
/compound-ml:ml-explore data/support_tickets.csv
/compound-ml:ml-explore logs/
```

Pass a file (CSV, JSON, Parquet) or a directory of text files (.txt, .md).

**What you get back:**

- Shape (rows × columns)
- Column-by-column profile — distributions for numbers, top values for text, date ranges
- Data quality summary — missing values, type issues
- Suggested next steps based on what the data looks like

**Tips:**
- This is the lowest-friction skill — it only needs pandas. No embeddings, no API keys.
- Files over 50,000 rows are automatically sampled. The output notes when this happens.
- If matplotlib is installed, you also get distribution charts.

---

### 2. Cluster — Find Natural Groups

**What it does:** Groups similar items together and labels each group in plain language. Think customer segments, topic categories, or behavior patterns — discovered automatically from the data.

**When to use it:** You want to find segments, categories, or themes in your data without defining them upfront.

**How to run it:**

```
/compound-ml:ml-cluster data/customers.csv "find customer segments"
/compound-ml:ml-cluster docs/ "group these documents by topic"
```

The objective is optional but improves labeling. Without it, the skill finds the most natural groupings.

**What you get back:**

- Number of groups found
- For each group: a plain-language label, description, size, representative examples, and distinguishing characteristics
- Quality indicator (good/fair/poor separation)
- Suggested next steps

**Tips:**
- **Text data quality depends on your embedding provider.** The skill auto-detects what's available:
  1. sentence-transformers (if installed) — good quality, free, runs locally
  2. TF-IDF fallback — always available, lower quality for text
- **Numeric data doesn't need embeddings** — features are used directly.
- Results are checkpointed. If the process is interrupted, it resumes from the last completed phase (checkpoints last 24 hours).

---

### 3. Anomalies — Flag Unusual Items

**What it does:** Finds items that are unusual compared to the rest of the dataset, then explains in plain language why each one stands out — with specific values and comparisons, not just "this is an anomaly."

**When to use it:** You want to find outliers, suspicious entries, or items that don't fit the pattern.

**How to run it:**

```
/compound-ml:ml-anomalies data/transactions.csv "flag suspicious transactions"
/compound-ml:ml-anomalies logs/api_responses.json "find unusual responses"
```

The context is optional but helps frame the explanations.

**What you get back:**

- Count and percentage of unusual items
- For each anomaly: a confidence level, a plain-language explanation referencing specific data values, and key differences from typical items
- Summary of whether anomalies share patterns or are isolated
- Suggested next steps

**How it works under the hood:** Two independent detection methods (Isolation Forest and Local Outlier Factor) each vote on what's unusual. Items flagged by both methods rank highest — this consensus approach reduces false positives.

**Tips:**
- Same embedding provider cascade as clustering (sentence-transformers > TF-IDF).
- Datasets under 10 rows skip statistical methods entirely and use reasoning-based assessment instead.
- Results cap at 20 anomalies (or 5% of the dataset, whichever is smaller) to keep the output actionable.
- Checkpointing works the same as clustering.

---

### 4. RAG — Ask Questions About Your Documents

**What it does:** Builds a searchable index from a collection of documents, then answers questions grounded in those documents with source citations. Two modes: `build` (index your docs) and `query` (ask questions).

**When to use it:** You have a collection of documents and want to ask natural-language questions that get answered from the content — not from the AI's general knowledge.

**How to run it:**

```
# Index your documents (one-time)
/compound-ml:ml-rag build docs/knowledge-base/

# Ask questions (repeatable)
/compound-ml:ml-rag query "How do we handle authentication timeouts?"
/compound-ml:ml-rag query "What are the API rate limits?"
```

If you run `/compound-ml:ml-rag` without a mode and an index already exists, it enters query mode automatically.

**What you get back:**

- **Build mode:** Confirmation of how many documents and chunks were indexed
- **Query mode:** A direct answer with source citations (file name, section heading, and a relevant quote from each source)

**Important — embedding requirement:** Unlike the other skills, RAG requires sentence-transformers. TF-IDF is not sufficient for retrieval quality. Make sure you have it installed: `pip install sentence-transformers`

**Tips:**
- Supported document formats: `.md`, `.txt`, `.text`, `.pdf`, `.html`, `.rst`
- Documents are split into chunks at heading and paragraph boundaries with overlap for context continuity.
- Retrieval uses both meaning-based search (vector) and keyword search (BM25) for better recall. Install `rank-bm25` for hybrid search; without it, vector-only search is used.
- The index is stored locally in `.ml-rag-index/`. Rebuild it if your documents change.
- For large document sets using API embeddings, cost is estimated before indexing begins.

---

### 5. Analyze — Full Pipeline in One Command

**What it does:** Runs an end-to-end analysis: profiles your data, selects the right methods, clusters, detects anomalies, reviews results for quality, and generates a comprehensive report. This orchestrates the same techniques available in the individual skills.

**When to use it:** You want to go from raw data to a complete insights report without deciding which individual skills to run.

**How to run it:**

```
/compound-ml:ml-analyze data/support_tickets.csv "find patterns in these support tickets"
/compound-ml:ml-analyze data/user_events.json
```

The objective is optional. Without one, it runs a general analysis (profile + cluster + anomaly detection).

**What you get back:**

A full markdown report with:
- **Executive summary** — the most important findings in 3-5 sentences
- **Data overview** — what the data contains and any quality issues
- **Groups discovered** — if clustering ran, each group with labels and examples
- **Unusual items** — if anomaly detection ran, each anomaly with explanations
- **Methodology** — brief note on what methods were used (for reproducibility)
- **Recommended next steps** — what to investigate further

The report is saved to `.ml-checkpoints/ml-analyze/<timestamp>/report.md` and displayed directly.

**Tips:**
- The skill automatically picks which analyses to run based on your data type and objective. Text data gets clustering + anomaly detection by default. Numeric data does too. If you just want a profile, say "explore this data" or use `ml-explore` directly.
- Results go through an automated quality review before the report is generated. If something looks off (degenerate clusters, suspicious anomaly counts), the pipeline adjusts and retries.
- Checkpointing lets you resume if interrupted. Each phase saves before proceeding.
- For large datasets, expect the full pipeline to take several minutes — embedding generation and UMAP are the main bottlenecks.

---

## Which Skill Should I Use?

| I want to... | Use this | Needs embeddings? | Notes |
|---|---|---|---|
| Understand what's in my data | `ml-explore` | No | Always a good first step |
| Find groups, segments, or topics | `ml-cluster` | For text data | Numeric data works without embeddings |
| Spot outliers or suspicious entries | `ml-anomalies` | For text data | Two methods vote for consensus |
| Ask questions about documents | `ml-rag` | **Yes (required)** | TF-IDF is not sufficient for RAG |
| Get a complete analysis in one pass | `ml-analyze` | For text data | Orchestrates explore + cluster + anomalies |
| I'm not sure yet | `ml-explore` | No | The output suggests what to try next |

## Choosing an Embedding Provider

Embeddings matter when you're working with text data. They convert text into numerical representations that capture meaning — similar text gets similar numbers.

| Provider | Quality | Cost | Setup |
|---|---|---|---|
| sentence-transformers | Good | Free, runs locally | `pip install sentence-transformers` (~100MB model download on first use) |
| TF-IDF (fallback) | Basic | Free | Always available via scikit-learn |

**Numeric data does not need embeddings** — the numbers are used directly after scaling.

**RAG requires sentence-transformers.** The other skills fall back to TF-IDF gracefully, but RAG cannot — retrieval quality with TF-IDF is too low.

## Finding Data to Explore

Need a dataset to try this on? [Kaggle Datasets](https://www.kaggle.com/datasets) hosts thousands of free, downloadable datasets across every domain. Search for a topic you're interested in, download a CSV, and point any skill at it.

### Recommended Kaggle Datasets

These datasets work well with compound-ml's unsupervised, foundation-model-powered approach:

**[Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)** — Best starting point. ~170k tracks with rich audio features (danceability, energy, valence, tempo, genre). Try `ml-cluster` to discover natural music genres without labels, `ml-anomalies` to find outlier tracks with unusual feature combinations, or `ml-analyze` for a full pipeline run.

**[Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)** — 2,240 customers with demographics, spending habits, and campaign responses. Great for `ml-cluster` (find customer segments) and `ml-anomalies` (flag unusual purchasing patterns).

**[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** — 284k transactions with 30 anonymized features. Highly imbalanced (0.17% fraud). Ideal for `ml-anomalies` — see if unsupervised detection can surface the fraudulent transactions without labels.

**[BBC News Articles](https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification)** — 2,225 articles across 5 categories. Good for `ml-cluster` (topic discovery from text) and `ml-rag` (build a searchable news knowledge base).

**[Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)** — 1,600 wines with physicochemical measurements. Small, numeric, and fast — good for a first walkthrough of `ml-explore` → `ml-cluster` → `ml-anomalies`.

Start with the Spotify Tracks dataset — it's the right size, has features that embed well, and gives every skill something meaningful to work with.

## Checkpointing and Resuming

All skills except `ml-explore` save progress to `.ml-checkpoints/<skill-name>/<timestamp>/` after each phase. If a run is interrupted — network error, timeout, you close the terminal — the skill checks for recent checkpoints (less than 24 hours old) on the next run and offers to resume.

You don't need to manage checkpoints manually. They're cleaned up naturally as they age past 24 hours.
