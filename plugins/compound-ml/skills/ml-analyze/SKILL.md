---
name: ml-analyze
description: "Run an end-to-end analysis on a dataset: profile, cluster, detect anomalies, and generate a plain-language report. Use when the user wants a comprehensive analysis, says 'analyze this data', 'find patterns', 'what can you tell me about this dataset', or wants to go from raw data to actionable insights."
argument-hint: "<file-path> [objective in natural language]"
---

# End-to-End Data Analysis

The flagship analysis skill. Takes a dataset and a natural language objective, automatically selects and runs the appropriate analysis methods, reviews the results for quality, and generates a comprehensive plain-language report.

This skill orchestrates the same techniques available in `ml-explore`, `ml-cluster`, and `ml-anomalies`, but in a single automated pipeline. Use the individual skills for more focused, interactive work.

## Input

- **File path** (required): CSV, JSON, Parquet file, or directory of text files
- **Objective** (optional): What to learn from the data in plain language (e.g., "understand my customer base", "find patterns in these support tickets", "what topics are in these documents?")

If no objective is provided, run a general analysis: profile, cluster, and check for anomalies.

## Workflow

### Phase 1: Environment Check and Data Profiling

Use `uv run python3` for all Python calls in this skill.

Check core packages (pandas, sklearn). Report and stop if missing.

Check optional packages (umap, hdbscan, sentence-transformers, matplotlib) and note which are available — this affects which analysis methods can run.

Load and profile the data:
- Shape, column types, distributions, missing values
- Determine data type: primarily text, numeric, or mixed
- For large datasets (>50K rows), sample for profiling but note the full size
- Identify potential analysis angles based on the data structure

Write profile to checkpoint: `.ml-checkpoints/ml-analyze/<timestamp>/profile.json`

Report a brief profile summary to the user before proceeding:
> "Loaded [filename]: [N] rows x [M] columns. [Brief description of data type and notable features]. Planning analysis approach..."

### Phase 2: Select Analysis Approach

Based on the data profile and the user's objective, decide which analyses to run. This is inline reasoning — not a separate agent call.

**Decision framework:**

| Data type | Objective signals | Analysis to run |
|-----------|------------------|-----------------|
| Text-heavy | "topics", "themes", "categories", "group" | Clustering with embedding |
| Text-heavy | "unusual", "outlier", "suspicious", "different" | Anomaly detection with embedding |
| Text-heavy | No specific objective | Clustering + anomaly detection |
| Numeric | "segments", "groups", "types" | Clustering on features |
| Numeric | "unusual", "outlier", "anomaly" | Anomaly detection on features |
| Numeric | No specific objective | Clustering + anomaly detection |
| Mixed | Any | Embed text + scale numeric, run both |
| Any | "explore", "profile", "understand" | Extended profiling (skip clustering/anomalies) |

Always run profiling. Add clustering, anomaly detection, or both based on the table above.

Report the plan:
> "Analysis plan: [1] Profile the data, [2] Find natural groups using clustering, [3] Flag unusual items. Proceeding..."

### Phase 3: Generate Representations (if clustering or anomaly detection selected)

**First, check the shared embedding cache** (see AGENTS.md "Embedding Cache" section). If a cached representation exists for this file, load it and skip to Phase 4. This is especially valuable for ml-analyze since users often run ml-explore first.

Follow the same embedding/representation logic as `ml-cluster` Phase 2:
- Detect embedding provider (sentence-transformers > TF-IDF fallback)
- For text: generate embeddings using sentence-transformers if available
- For numeric: scale features with StandardScaler
- For mixed: concatenate embedded text with scaled numeric

Write to the shared embedding cache (see AGENTS.md) and to checkpoint: `.ml-checkpoints/ml-analyze/<timestamp>/representations.npy`

Use `timeout: 600000` for embedding generation on large datasets.

### Phase 4: Execute Analysis

**When both clustering and anomaly detection are selected, run them in parallel** using the Agent tool. Launch two agents simultaneously in a single message — one for each analysis. Both read from the same representations checkpoint written in Phase 3.

**Clustering agent** (if selected):

Launch an Agent with a prompt that includes the file path, checkpoint directory, representations path, data profile, objective, and available packages. The agent should follow `ml-cluster` Phases 3-5:
1. Dimensionality reduction with UMAP only if available AND representations have >50 dimensions (skip for numeric-only data or low-dimensional features)
2. Cluster with HDBSCAN or KMeans
3. Evaluate quality (silhouette score)
4. Sample representatives per cluster
5. Generate plain-language labels via LLM

Write to checkpoint: `.ml-checkpoints/ml-analyze/<timestamp>/clusters.json`

**Anomaly detection agent** (if selected):

Launch an Agent with a prompt that includes the file path, checkpoint directory, representations path, data profile, objective, and available packages. The agent should follow `ml-anomalies` Phases 3-4:
1. Run Isolation Forest and Local Outlier Factor
2. Compute consensus scores
3. Select top anomalies
4. Gather nearest normal neighbors for context

Write to checkpoint: `.ml-checkpoints/ml-analyze/<timestamp>/anomalies.json`

Use `timeout: 600000` for both agents. If only one analysis is selected, run it directly without spawning an agent.

Wait for both agents to complete before proceeding to Phase 5.

### Phase 5: Quality Review

Invoke the `compound-ml:review:ml-output-reviewer` agent to check results for quality issues:
- Degenerate clusters (all items in one group, or every item its own group)
- Suspicious anomaly counts (0% flagged or >30% flagged)
- Results that contradict the data profile (e.g., claiming "no patterns" in obviously structured data)

If the reviewer flags issues, adjust and re-run the affected analysis step with different parameters. If issues persist after one retry, include the quality concerns in the report.

### Phase 6: Generate Report

Produce a comprehensive markdown report. The report must be readable by someone with no ML background — all findings explained in plain language.

**Report structure:**

```markdown
# Analysis Report: [filename]

**Date:** [YYYY-MM-DD]
**Objective:** [user's objective or "General analysis"]
**Data:** [N] rows x [M] columns

## Executive Summary

[3-5 sentences capturing the most important findings. Lead with actionable insights, not methodology.]

## Data Overview

[Brief profile: what the data contains, data quality notes, any sampling applied]

## Findings

### Groups Discovered
[If clustering was run — describe each group with labels, sizes, descriptions, and representative examples. Use the same format as ml-cluster Phase 6.]

### Unusual Items
[If anomaly detection was run — describe top anomalies with explanations. Use the same format as ml-anomalies Phase 6.]

## Methodology

[Brief plain-language description of what methods were used and why. Mention embedding type, clustering algorithm, anomaly detectors. Keep this to 2-3 sentences — it's here for reproducibility, not the main event.]

## Recommended Next Steps

[Actionable suggestions based on findings:
- What to investigate further
- Which groups or anomalies deserve attention
- What additional data might help
- Which individual skills to use for deeper dives]
```

Write the report to `.ml-checkpoints/ml-analyze/<timestamp>/report.md` and also display it directly to the user.

## Checkpointing and Resume

On start, check for recent checkpoints (<24h) in `.ml-checkpoints/ml-analyze/`:
- If found, report what was completed and ask: "Found a previous analysis in progress. Resume from [Phase N], or start fresh?"
- If resuming, load checkpoints and continue from the next incomplete phase

## Error Handling

- **Missing core packages:** Report install instructions and stop
- **No embedding provider for text data:** Fall back to TF-IDF with quality note in report
- **All analysis methods fail:** Generate a report focused on the data profile with quality issues noted and cleaning suggestions
- **Objective doesn't match data:** Report the mismatch in the executive summary and suggest alternative analyses
- **Timeout on large computation:** Report which phase timed out, suggest reducing dataset size or installing missing packages for faster execution

## Reference Files

- `references/workflow-guide.md` — Overview of the analysis pipeline for curious users
