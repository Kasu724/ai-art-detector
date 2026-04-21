# Data Layout

Raw datasets are intentionally not stored in Git. Use one of these layouts:

```text
data/raw/human/<image files>
data/raw/ai/<image files>
```

or:

```text
data/raw/<source_name>/human/<image files>
data/raw/<source_name>/ai/<image files>
```

The preparation command converts the raw files into a reproducible manifest and summary file under `data/processed/`.

Built-in adapter:
- folder-based ingestion via `src/ai_art_detector/data/adapters.py`

Smoke-data helper:
- `python scripts/generate_smoke_dataset.py`
- useful for pipeline checks only, not for reporting real detector quality
