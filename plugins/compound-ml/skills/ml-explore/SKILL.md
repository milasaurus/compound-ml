---
name: ml-explore
description: "Profile a dataset and narrate findings in plain language. Use when the user wants to understand their data, explore a CSV/JSON/Parquet file, get a data summary, or says 'explore this data', 'what does this dataset look like', or 'profile my data'."
argument-hint: "<file-path or directory>"
---

# Explore and Profile a Dataset

Automated exploratory data analysis that profiles a dataset and narrates findings in plain language. No ML expertise required — the analysis runs automatically and results are explained without jargon.

This skill does NOT generate embeddings or run ML algorithms. It profiles the data structure, distributions, and quality. For clustering, anomaly detection, or deeper analysis, use `ml-cluster`, `ml-anomalies`, or `ml-analyze`.

## Input

The user provides a file path or directory as an argument:
- **Tabular data:** CSV, JSON, or Parquet file
- **Text corpus:** A directory of text/markdown files (each file treated as one document)

If no argument is provided, scan the working directory for data files and ask which one to profile.

## Workflow

### Phase 1: Environment Check

Verify pandas is available:

```bash
uv run python3 -c "import pandas; print(f'pandas {pandas.__version__}')"
```

Use `uv run python3` for all Python calls in this skill.

If pandas is not installed, report:

> pandas is required for data exploration. Install it with:
> ```
> uv pip install pandas
> ```
> See `references/setup.md` for full setup instructions.

Do not proceed until the environment check passes.

### Phase 2: Load and Profile Data

#### For tabular files (CSV, JSON, Parquet)

Run a Python script via `Bash(uv run python3 -c "...")` that:

1. Loads the file with pandas (detect format from extension)
2. If the file has more than 50,000 rows, sample 50,000 rows and note this in output
3. Reports:
   - Shape (rows x columns)
   - Column names with data types
   - For numeric columns: min, max, mean, median, std dev, count of nulls
   - For text/object columns: count of unique values, most common values (top 5), count of nulls, average string length
   - For datetime columns: min date, max date, count of nulls
   - Overall missing value summary
   - First 5 rows as a sample

Print all output as structured text that the LLM can interpret.

#### For text directories

Run a Python script that:

1. Lists all `.txt`, `.md`, `.text` files in the directory (non-recursive by default)
2. Reports:
   - Total document count
   - Average, min, and max document length (in characters and words)
   - Most common words across the corpus (top 20, excluding common stop words)
   - Sample of first 3 document previews (first 200 characters each)

### Phase 3: Visualize (Optional)

Attempt to generate distribution visualizations:

1. Check if matplotlib is available: `uv run python3 -c "import matplotlib"`
2. If available, generate a summary visualization:
   - For tabular data: histograms of numeric columns (save to a temp PNG file)
   - For text data: bar chart of document length distribution
3. If matplotlib is not available, skip visualization silently — the narrative summary is sufficient

### Phase 4: Narrate Findings

Using the profiling output from Phase 2, generate a plain-language narrative summary. The summary must:

- Lead with the most important finding ("Your dataset has 15,000 customer support tickets spanning 18 months")
- Highlight data quality issues ("23% of the 'category' column is empty — this may affect grouping")
- Note interesting patterns ("Transaction amounts are heavily right-skewed — most are under $50 but a few exceed $10,000")
- Suggest next steps based on what was found:
  - Text-heavy data → suggest `ml-cluster` for topic discovery or `ml-anomalies` for outlier detection
  - Numeric data with potential groups → suggest `ml-cluster`
  - Data with potential outliers → suggest `ml-anomalies`
  - Collection of documents → suggest `ml-rag` for building a searchable knowledge base
- Use no ML jargon. Terms like "right-skewed" should be rephrased as "most values are low but a few are very high"

## Output Format

Present findings as a structured narrative:

```
## Data Profile: [filename]

**Shape:** [rows] rows x [columns] columns
**Sampled:** [Yes/No — if sampled, note original size]

### Summary
[2-3 sentence plain-language overview of the dataset]

### Column Details
[Table or list of column profiles]

### Data Quality
[Missing values, type issues, notable patterns]

### Suggested Next Steps
[1-3 actionable suggestions based on findings]
```

## Error Handling

- **Unsupported format** (e.g., .xlsx without openpyxl): Report the format and suggest the install command for the reader package
- **Empty file or single row:** Report "Not enough data for meaningful profiling" with the actual row count
- **File not found:** Report the error and suggest checking the path
- **Encoding errors:** Try UTF-8 first, then latin-1 fallback, then report the issue
- **Memory issues on very large files:** If loading fails, retry with `nrows=50000` parameter

## Reference Files

- `references/setup.md` — Full environment setup guide
- `references/data-formats.md` — Supported formats and column type handling
