# How the Analysis Pipeline Works

## Overview

The `ml-analyze` skill runs a multi-step analysis pipeline that automatically adapts to your data and objective. Here's what happens at each step.

## Step 1: Profile Your Data

Before any analysis, the pipeline reads your data and builds a profile:
- How many rows and columns
- What types of data (text, numbers, dates)
- Data quality (missing values, duplicates)
- Basic statistics for each column

This profile guides which analysis methods to use.

## Step 2: Choose Analysis Methods

Based on your data type and objective, the pipeline selects the right combination:

- **Text data** → Convert to numerical representations using AI embeddings, then cluster and/or detect anomalies
- **Numeric data** → Use the numbers directly for clustering and/or anomaly detection
- **Mixed data** → Combine both approaches

The specific methods chosen depend on what packages are installed and what embedding providers are available.

## Step 3: Represent the Data

For the analysis algorithms to work, every item in your data needs a numerical representation:

- **Best quality**: OpenAI embeddings or sentence-transformers capture the meaning of text
- **Acceptable quality**: TF-IDF captures word patterns (available even without API keys)
- **Numeric data**: Used directly after scaling to make all features comparable

## Step 4: Run Analysis

### Clustering

Finds natural groups by looking for items that are similar to each other. The algorithm picks the number of groups automatically — you don't need to specify how many groups to look for. Each group gets a plain-language label describing what its members have in common.

### Anomaly Detection

Identifies items that are unusual compared to the rest of the dataset. Two different methods vote on what's unusual, and items flagged by both methods are considered the strongest anomalies. Each anomaly gets an explanation of why it stands out.

## Step 5: Quality Check

The results are automatically reviewed for common problems:
- All items in one group (means the data is too uniform for clustering)
- Too many anomalies flagged (means the threshold is too sensitive)
- Results that don't make sense given the data

If problems are found, the analysis adjusts and retries.

## Step 6: Generate Report

All findings are compiled into a structured report with:
- An executive summary of the most important insights
- Detailed findings for each analysis method
- Plain-language explanations throughout
- Recommended next steps

## Checkpointing

Each step saves its results to disk. If the analysis is interrupted (network error, timeout, etc.), it can resume from where it left off rather than starting over. Checkpoints are kept for 24 hours.

## What This Pipeline Does NOT Do

- **Train models** — All models used are pre-trained
- **Require labeled data** — Everything is unsupervised (no "right answers" needed)
- **Require ML expertise** — You describe what you want in plain language
- **Modify your data** — Your original files are never changed
