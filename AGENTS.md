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

**ML workflow skills** use `ml-` prefix in their directory name:
- `ml-explore` - Profile and narrate dataset characteristics
- `ml-cluster` - Find natural groups with plain-language labels
- `ml-anomalies` - Detect and explain unusual items
- `ml-rag` - Build and query retrieval-augmented generation pipelines
- `ml-analyze` - End-to-end analysis from data to insights report

Slash commands appear as `compound-ml:<skill-directory-name>` (e.g., `compound-ml:ml-explore`).

## Agent Namespacing

Agents use fully-qualified names: `compound-ml:<category>:<agent-name>`

Examples:
- `compound-ml:research:ml-literature-researcher`
- `compound-ml:review:ml-output-reviewer`

## Skill Compliance Checklist

### YAML Frontmatter (Required)

- [ ] `name:` present and matches directory name (lowercase-with-hyphens)
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

- Use `Bash(python3 -c "...")` for inline Python scripts
- Use `timeout: 600000` (10 minutes) for long-running operations (UMAP, large embeddings, HDBSCAN)
- Break large computations into stages with intermediate file I/O
- Check for required packages at startup with `python3 -c "import X"` before using them
- If a package is missing, report the install command rather than auto-installing

## Embedding Provider Detection

Skills that need embeddings follow this detection cascade:
1. OpenAI API — check for `OPENAI_API_KEY` environment variable
2. Local sentence-transformers — check `python3 -c "import sentence_transformers"`
3. TF-IDF fallback — always available via sklearn (not viable for RAG)

## Checkpointing

File-based checkpoints in `.ml-checkpoints/<skill-name>/<timestamp>/` as JSON/CSV.
Skills check for recent checkpoints (<24h) on start and offer to resume.
