# Compound ML Plugin

ML workflow tools that leverage foundation models so you don't have to be an ML expert. Get insights from your data using plain language — the agent handles the complexity.

## Who This Is For

People who want to explore, cluster, and analyze data using foundation models without needing ML training expertise or labeled datasets. All workflows use pre-trained models and unsupervised techniques.

## Skills

| Skill | Description |
|-------|-------------|
| `ml-explore` | Profile a dataset and narrate findings in plain language |
| `ml-cluster` | Find natural groups in data with plain-language labels |
| `ml-anomalies` | Detect unusual items and explain why they stand out |
| `ml-rag` | Build and query a retrieval-augmented generation pipeline |
| `ml-analyze` | End-to-end analysis: data in, insights report out |

## Agents

Agents are specialized subagents invoked by skills — not called directly.

| Agent | Description |
|-------|-------------|
| `ml-literature-researcher` | Recommend ML techniques for a given analytical objective |
| `ml-output-reviewer` | Review analysis outputs for quality problems |

## Setup

### Requirements

- Python 3.10+
- Claude Code

### Recommended Packages

```bash
pip install pandas scikit-learn umap-learn hdbscan matplotlib
```

For embedding-based workflows (higher quality clustering/anomaly detection):
```bash
pip install sentence-transformers
```

For RAG pipelines:
```bash
pip install chromadb rank-bm25
```

See `skills/ml-explore/references/setup.md` for detailed setup instructions.

## Installation

```bash
claude /plugin install compound-ml
```

## Not In Scope

This plugin does not train or fine-tune models. It uses pre-trained foundation models exclusively. For model training, see dedicated ML platforms.

## License

MIT
