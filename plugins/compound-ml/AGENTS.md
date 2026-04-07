# Plugin Instructions

These instructions apply when working under `plugins/compound-ml/`.

# Compound ML Plugin Development

## Directory Structure

```
agents/
├── research/         # Research and technique recommendation agents
└── review/           # Output quality review agents

skills/
├── ml-*/             # Core ML workflow skills
└── */                # All other skills
```

## Skill Naming Convention

**ML workflow skills** use `ml-` prefix in their skill name:
- `ml-explore` - Profile and narrate dataset characteristics
- `ml-cluster` - Find natural groups with plain-language labels
- `ml-anomalies` - Detect and explain unusual items
- `ml-rag` - Build and query retrieval-augmented generation pipelines
- `ml-analyze` - End-to-end analysis from data to insights report

Slash commands appear as `compound-ml:ml-explore`, `compound-ml:ml-cluster`, etc.

## Agent Namespacing

Agents use fully-qualified names: `compound-ml:<category>:<agent-name>`

Examples:
- `compound-ml:research:ml-literature-researcher`
- `compound-ml:review:ml-output-reviewer`

## Skill Compliance Checklist

### YAML Frontmatter (Required)

- [ ] `name:` present and uses colon-namespaced format (e.g., `ml-explore`)
- [ ] `description:` present and describes what it does and when to use it
- [ ] `description:` value is quoted (single or double) if it contains colons

### Reference File Inclusion

- [ ] Use backtick paths for reference files: `references/filename.md`
- [ ] Do NOT use markdown links like `[filename.md](./references/filename.md)`
- [ ] Use `@` inline only for small structural files under ~150 lines that the skill cannot function without

### Writing Style

- [ ] Use imperative/infinitive form (verb-first instructions)
- [ ] Avoid second person ("you should") — use objective language
- [ ] All user-facing output must be in plain language — no ML jargon unless the user requests technical details

## Python Execution Conventions

- **Always use `uv run` to execute Python.** Never call bare `python3` — always use `uv run python3`. This automatically uses the project's `.venv` without manual activation.
- For short scripts (under ~20 lines with no nested quotes), use `Bash(uv run python3 -c "...")`
- For longer or complex scripts, write the script to a temp file first, then execute it. This avoids quoting and escaping issues with heredocs and f-strings:
  ```
  Write /tmp/ml_task.py  (the Python code)
  Bash(uv run python3 /tmp/ml_task.py)
  ```
  Always use `/tmp/ml_*.py` as the naming convention so they're naturally cleaned up
- Use `timeout: 600000` (10 minutes) for long-running operations (UMAP, large embeddings, HDBSCAN)
- Break large computations into stages with intermediate file I/O
- Check for required packages at startup with `uv run python3 -c "import X"` before using them
- If a package is missing, report the install command rather than auto-installing
- When selecting columns by dtype, use `select_dtypes(include=["object", "str"])` for text columns — pandas 3.x requires explicit `"str"` inclusion

## Embedding Provider Detection

Skills that need embeddings follow this detection cascade:
1. Local sentence-transformers — check `uv run python3 -c "import sentence_transformers"`
2. TF-IDF fallback — always available via sklearn (not viable for RAG)

## Embedding Cache

Skills that generate embeddings or scaled representations must use a shared cache at `.ml-checkpoints/_embeddings/` to avoid redundant computation across skills.

**Cache key:** `<filename>_<row_count>_<provider>_<column_hash>.npy` where:
- `filename` — the source data file name (without path)
- `row_count` — number of rows in the dataset
- `provider` — `sentence-transformers`, `tfidf`, or `numeric`
- `column_hash` — first 8 chars of MD5 hash of sorted column names used

**Before generating representations**, check for a matching cache file:

```python
import hashlib, os, numpy as np
cache_dir = ".ml-checkpoints/_embeddings"
col_hash = hashlib.md5(",".join(sorted(columns)).encode()).hexdigest()[:8]
cache_key = f"{filename}_{row_count}_{provider}_{col_hash}.npy"
cache_path = os.path.join(cache_dir, cache_key)
if os.path.exists(cache_path):
    representations = np.load(cache_path)
    print(f"Loaded cached embeddings from {cache_path}")
```

**After generating representations**, save to cache:

```python
os.makedirs(cache_dir, exist_ok=True)
np.save(cache_path, representations)
```

Cache files have no expiry — they remain valid as long as the source data hasn't changed (keyed by filename + row count + columns). The cache directory is gitignored via `.ml-checkpoints/`.

## Checkpointing

File-based checkpoints in `.ml-checkpoints/<skill-name>/<timestamp>/` as JSON/CSV.
Skills check for recent checkpoints (<24h) on start and offer to resume.
