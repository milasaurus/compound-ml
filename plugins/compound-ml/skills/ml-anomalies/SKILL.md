---
name: ml-anomalies
description: "Find unusual items in a dataset and explain why they stand out. Use when the user wants to detect outliers, flag suspicious entries, find anomalies, or says 'find outliers', 'what looks unusual', 'flag suspicious', or 'detect anomalies'."
argument-hint: "<file-path> [context in natural language]"
---

# Detect and Explain Anomalies

Find items in a dataset that are unusual compared to the rest, and explain in plain language why each one stands out. No ML expertise required — detection methods, thresholds, and explanations are all handled automatically.

## Input

- **File path** (required): CSV, JSON, Parquet file, or directory of text files
- **Context** (optional): Natural language description of what to look for (e.g., "flag suspicious transactions", "find outlier documents", "spot unusual customer behavior")

If no context is provided, detect the most statistically unusual items across all available features.

## Workflow

### Phase 1: Environment Check and Data Loading

Check required packages:

```bash
python3 -c "import pandas; import sklearn; print('Core packages available')"
```

If pandas or sklearn are missing, report install instructions and stop.

Load and profile the data. Identify:
- Data type (text, numeric, mixed)
- Number of rows and columns
- Any quality issues

Write data profile to checkpoint: `.ml-checkpoints/ml-anomalies/<timestamp>/profile.json`

**Small dataset handling:** If the dataset has fewer than 10 rows, skip statistical methods entirely. Instead, present all items to the LLM and ask it to reason about which (if any) seem unusual compared to the others. Report: "With only [N] items, statistical anomaly detection isn't reliable. Here's a reasoning-based assessment instead."

### Phase 2: Generate Representations

Follow the same representation strategy as `ml-cluster`:

**For text data:** Detect embedding provider (sentence-transformers > TF-IDF fallback).

**For numeric data:** Use numeric columns directly with `StandardScaler` normalization.

**For mixed data:** Embed text columns and concatenate with scaled numeric features.

Write representations to checkpoint: `.ml-checkpoints/ml-anomalies/<timestamp>/representations.npy`

### Phase 3: Run Anomaly Detectors

Run two complementary methods for consensus scoring:

**Isolation Forest:**

```python
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination='auto', random_state=42)
iso_scores = iso.decision_function(representations)  # lower = more anomalous
iso_labels = iso.predict(representations)  # -1 = anomaly, 1 = normal
```

**Local Outlier Factor:**

```python
from sklearn.neighbors import LocalOutlierFactor
n_neighbors = min(20, max(5, len(data) // 10))
lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination='auto')
lof_labels = lof.fit_predict(representations)  # -1 = anomaly, 1 = normal
lof_scores = lof.negative_outlier_factor_  # more negative = more anomalous
```

**Consensus scoring:**

Normalize both score arrays to 0-1 range (0 = most normal, 1 = most anomalous). Average them for a consensus score. Items flagged by both methods rank higher.

```python
# Normalize scores to 0-1
iso_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
iso_norm = 1 - iso_norm  # flip so higher = more anomalous
lof_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min())
lof_norm = 1 - lof_norm  # flip so higher = more anomalous

consensus = (iso_norm + lof_norm) / 2
both_flagged = (iso_labels == -1) & (lof_labels == -1)
```

Write scores to checkpoint: `.ml-checkpoints/ml-anomalies/<timestamp>/scores.json`

### Phase 4: Select and Contextualize Anomalies

Select the top anomalies:
- Items flagged by both methods (consensus anomalies) are included first
- Then items flagged by either method, ranked by consensus score
- Cap at 20 anomalies maximum (or 5% of the dataset, whichever is smaller)

If no items are flagged by either method, report:
> "Nothing unusual detected in this dataset. The items appear consistent with each other."

For each anomaly, gather context:
1. The anomaly's full data row
2. Its 3 nearest normal neighbors (by distance in representation space)
3. The overall distribution statistics for each feature

### Phase 5: Generate Explanations

For each anomaly (or batch of up to 5 anomalies at a time), provide the LLM with:
- The anomalous item's data
- Its nearest normal neighbors for comparison
- The overall distribution statistics
- The user's context (if provided)

Ask the LLM to explain WHY each item is unusual, referencing specific data values:

- Good: "This transaction stands out because the amount ($8,500) is 47x the median ($180) for this account, and it occurred at 3:15 AM — 95% of this account's transactions happen between 9 AM and 6 PM."
- Bad: "This is an anomaly." (too vague)
- Bad: "The isolation forest score is -0.82." (jargon)

Frame explanations in the user's context when provided. For "flag suspicious transactions", focus on what makes each transaction suspicious. For "find outlier documents", explain how each document differs from the collection's themes.

### Phase 6: Present Results

Format the output as:

```
## Anomaly Detection Results: [filename]

**Items analyzed:** [N]
**Unusual items found:** [M] ([X]% of dataset)
**Detection methods:** Isolation Forest + Local Outlier Factor (consensus)
**Representations:** [sentence-transformers / TF-IDF / numeric features]

### Most Unusual Items

#### 1. [Brief identifier — e.g., "Row 4,823" or "Transaction #TX-9981"]
**Anomaly confidence:** [High/Medium] (flagged by [both methods / one method])

[Plain-language explanation referencing specific values]

**Key differences from typical items:**
- [Specific comparison point 1]
- [Specific comparison point 2]

#### 2. [Brief identifier]
...

### Summary
[2-3 sentence overview: how many anomalies, what patterns they share, whether they suggest a systemic issue or are isolated cases]

### Suggested Next Steps
- [Contextual suggestions — e.g., "Review these 5 transactions with your fraud team" or "These documents may be miscategorized"]
```

## Checkpointing and Resume

Same pattern as `ml-cluster`: check for recent checkpoints (<24h) on start, offer to resume or start fresh. Each phase writes output before proceeding.

## Error Handling

- **sklearn not installed:** Report install instructions and stop (sklearn is required for both detectors)
- **No sentence-transformers for text data:** Fall back to TF-IDF with quality note
- **Dataset too small (<10 rows):** Use LLM-only reasoning instead of statistical methods
- **All items flagged as anomalous:** Report "the detection methods flagged an unusually high number of items — this usually means the data is very diverse rather than having specific outliers" and show only the top 10 by consensus score
- **Embedding API failure:** Save completed work to checkpoint, report error, suggest retry

## Reference Files

- `references/anomaly-detection-guide.md` — Plain-language explanation of how anomaly detection works
