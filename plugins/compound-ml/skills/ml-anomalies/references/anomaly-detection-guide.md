# How Anomaly Detection Works (Plain Language)

## What Is Anomaly Detection?

Anomaly detection finds items that don't fit the pattern of the rest of the data. Think of a credit card statement where every charge is $20-80 except one for $8,500 at 3 AM — that one stands out, and you'd want to know about it.

## How This Skill Does It

### Step 1: Represent the Data

Same as clustering — each item gets a numeric representation:

- **Text data** is converted to embeddings that capture meaning. A customer complaint about "product arrived broken" would be far from a complaint about "slow delivery" in this numeric space.
- **Numeric data** is scaled so no single column dominates. A $10,000 transaction shouldn't outweigh a timestamp just because the number is bigger.

### Step 2: Detect Anomalies (Two Methods)

Two independent methods each decide what's unusual:

- **Isolation Forest:** Picks random features and random split points, then measures how quickly each item can be separated from the rest. Normal items take many splits to isolate (they're in the crowd). Anomalies take very few splits (they're already standing apart).

- **Local Outlier Factor (LOF):** Looks at each item's neighborhood — how dense is the area around it compared to the areas around its neighbors? An item in a sparse region surrounded by items in dense regions is likely an outlier.

### Step 3: Consensus Scoring

Items flagged by both methods are the strongest anomalies. Items flagged by only one method are weaker signals. This reduces false positives — each method has different blind spots, so agreement between them is more reliable.

### Step 4: Explain Why

For each anomaly, the skill compares it to its nearest normal neighbors and the overall data distribution. Explanations reference specific values and comparisons, not scores or jargon.

## When Anomaly Detection Works Well

- Fraud detection in transaction data
- Finding mislabeled or miscategorized items
- Quality control — spotting defective products or bad data entries
- Identifying unusual documents in a collection
- Flagging outlier behavior in user activity logs

## When Anomaly Detection May Struggle

- Very small datasets (under 10 items) — not enough data for statistical patterns. The skill switches to reasoning-based assessment instead.
- Data where every item is genuinely unique (high-dimensional creative content)
- Data with many natural subgroups — items at group boundaries may be flagged as anomalies when they're just between-group transitions
- Datasets where the "anomalies" are actually a second population (consider clustering instead)

## Understanding Results

- **High confidence** (flagged by both methods): Very likely a genuine anomaly. Worth investigating.
- **Medium confidence** (flagged by one method): May be unusual or may be a borderline case. Use the explanation to judge.
- **No anomalies found:** The data is relatively uniform — items are consistent with each other.

## Improving Results

1. **Better embeddings**: Use sentence-transformers instead of TF-IDF for text data
2. **Feature selection**: Focus on columns most relevant to what "unusual" means in your context
3. **Provide context**: Adding a natural-language objective (e.g., "flag suspicious transactions") improves the explanations
4. **More data**: Statistical methods are more reliable with larger datasets
