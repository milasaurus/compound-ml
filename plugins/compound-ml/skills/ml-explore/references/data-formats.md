# Supported Data Formats

## Tabular Files

| Format | Extension | Reader | Notes |
|--------|-----------|--------|-------|
| CSV | `.csv` | `pandas.read_csv()` | Auto-detects delimiter. Handles quoted fields. |
| JSON | `.json` | `pandas.read_json()` | Supports records, columns, and values orientations. |
| Parquet | `.parquet` | `pandas.read_parquet()` | Requires `pyarrow` or `fastparquet`. Most efficient for large files. |
| TSV | `.tsv` | `pandas.read_csv(sep='\t')` | Tab-separated variant of CSV. |

### Not Supported by Default

| Format | Extension | Required Package | Install Command |
|--------|-----------|-----------------|-----------------|
| Excel | `.xlsx` | openpyxl | `uv pip install openpyxl` |
| Excel (legacy) | `.xls` | xlrd | `uv pip install xlrd` |
| Feather | `.feather` | pyarrow | `uv pip install pyarrow` |
| HDF5 | `.h5`, `.hdf5` | tables | `uv pip install tables` |

## Text Corpora

When a directory path is provided, the skill scans for text files:

| Extension | Treatment |
|-----------|-----------|
| `.txt` | Plain text document |
| `.md` | Markdown document (treated as plain text for profiling) |
| `.text` | Plain text document |

Each file is treated as one document. Subdirectories are not scanned by default.

## Column Type Detection

Pandas infers column types automatically. The profiling adapts based on detected type:

| Pandas Type | Profile Includes |
|-------------|-----------------|
| `int64`, `float64` | Min, max, mean, median, std dev, null count |
| `object` (text) | Unique count, top 5 values, null count, avg string length |
| `datetime64` | Min date, max date, null count |
| `bool` | True/False counts |
| `category` | Category list, value counts |

## Large File Handling

- Files over 50,000 rows are sampled (random 50,000 rows)
- The sampling is noted in the output so the user knows the profile is approximate
- For exact counts (nulls, uniques), the full file is used when possible via chunked reading
- Parquet files support efficient column-level statistics without loading the full dataset

## Encoding

Files are read with UTF-8 encoding by default. If that fails:
1. Retry with `latin-1` (ISO 8859-1) encoding
2. If both fail, report the encoding error and suggest the user specify the encoding
