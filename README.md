# AI Art Detector

Research-style image classifier for detecting whether an artwork image is more likely AI-generated or human-made.

The repository is designed as a serious engineering project rather than a single notebook:
- manifest-driven data preparation
- PyTorch transfer-learning training pipeline
- evaluation with threshold tuning, ROC-AUC, calibration, and error analysis
- FastAPI inference API
- Streamlit demo
- Docker support
- experiment configs for baseline and improved variants
- reproducible artifact layout and tests

## Motivation

AI-art detection looks deceptively simple if you only report accuracy on one curated dataset. In practice, it is a brittle distribution-shift problem:
- models can learn generator-specific shortcuts instead of general provenance cues
- train/test leakage can happen through source overlap, compression signatures, or shared post-processing
- a model that looks strong on one benchmark may fail on another image domain or generator family
- calibrated confidence is still not proof of authenticity

This repo is intentionally structured so those risks are visible:
- datasets are prepared into explicit manifests
- splits are saved and reproducible
- evaluation reports more than accuracy
- confidence thresholds and calibration are surfaced explicitly
- the README documents domain-shift and leakage concerns instead of hiding them

## What Is Included

### Core pipeline
- dataset preparation from a raw folder tree into a reproducible CSV manifest
- folder-based dataset adapter boundary in [src/ai_art_detector/data/adapters.py](src/ai_art_detector/data/adapters.py)
- PyTorch training loop with transfer learning and checkpointing
- held-out evaluation with:
  - accuracy
  - precision
  - recall
  - F1
  - ROC-AUC
  - average precision
  - confusion matrix
  - Brier score
  - expected calibration error
- threshold tuning on a validation split
- temperature-scaling calibration support
- error analysis artifact with top confident mistakes
- experiment comparison utility that writes JSON and Markdown summaries

### Product surfaces
- FastAPI API:
  - `GET /health`
  - `POST /predict`
  - `POST /predict-batch`
- Streamlit demo for interactive uploads
- single-image CLI inference
- ONNX export path

### Engineering / reproducibility
- config-driven experiments under `configs/`
- run artifacts grouped by stage under `artifacts/`
- tests for config loading, dataset preparation, metrics, and API behavior
- Dockerfile for the API and an additional demo Dockerfile
- `Makefile` and `scripts/` wrappers for common tasks
- smoke dataset generator for pipeline checks

## Experiment Lineup

The repository ships three experiment configs:

| Experiment | Config | Purpose | Backbone |
| --- | --- | --- | --- |
| Baseline | `configs/experiment.yaml` | Resume-grade transfer-learning baseline | `resnet18` |
| Improved | `configs/experiment_improved.yaml` | Stronger comparison run with heavier augmentation | `efficientnet_b0` |
| Smoke | `configs/experiment_smoke.yaml` | Fast CPU-only pipeline verification | `tiny_cnn` |

### Baseline experiment
- `resnet18`
- ImageNet pretrained weights
- moderate augmentation
- validation-driven checkpoint selection
- threshold tuning and calibration handled in evaluation

### Improved experiment
- `efficientnet_b0`
- slightly stronger augmentation
- longer training schedule
- intended as the first comparison point against the baseline

### Smoke experiment
- tiny custom CNN
- not a research result
- useful for CI-like checks, local sanity testing, and demos of the pipeline without pulling pretrained weights

## Repository Structure

```text
ai-art-detector/
|-- configs/
|   |-- experiment.yaml
|   |-- experiment_improved.yaml
|   |-- experiment_smoke.yaml
|   |-- data/
|   |-- model/
|   |-- train/
|   |-- eval/
|   `-- runtime/
|-- data/
|   |-- raw/
|   |-- interim/
|   `-- processed/
|-- artifacts/
|-- docker/
|-- scripts/
|-- src/ai_art_detector/
|   |-- api/
|   |-- data/
|   |-- demo/
|   |-- evaluation/
|   |-- inference/
|   |-- models/
|   |-- tracking/
|   |-- training/
|   `-- utils/
|-- tests/
|-- Dockerfile
|-- Makefile
`-- pyproject.toml
```

## Dataset Format

Raw datasets are expected outside Git and are prepared into a project manifest.

### Supported raw layouts

Simplest layout:

```text
data/raw/human/<image files>
data/raw/ai/<image files>
```

Preferred multi-source layout:

```text
data/raw/<source_name>/human/<image files>
data/raw/<source_name>/ai/<image files>
```

The multi-source layout is recommended because it makes source-aware inspection and leakage checks easier.

### What preparation produces

`prepare-data` scans the raw tree and writes:
- `data/processed/dataset_manifest.csv`
- `data/processed/preparation_summary.json`
- `artifacts/data_preparation/<timestamp>_<experiment>/resolved_config.yaml`
- `artifacts/data_preparation/<timestamp>_<experiment>/data_preparation_summary.json`

Manifest fields include:
- `sample_id`
- `path`
- `relative_path`
- `label`
- `label_name`
- `source`
- `split`
- `file_size_bytes`
- `sha256`
- `width`
- `height`
- `extension`

### Adapter layer

The current built-in adapter is the folder-based adapter in [src/ai_art_detector/data/adapters.py](src/ai_art_detector/data/adapters.py). If you want to integrate a public benchmark, Kaggle dataset, or a custom CSV-based corpus, this is the boundary to extend rather than rewriting the training code.

## Installation

Python 3.11+ is required.

### Minimal setup for data preparation and tests

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .[dev]
```

### Full setup for training, evaluation, API, demo, and ONNX

```bash
python -m pip install -e .[dev,ml,api,demo,onnx]
```

## Quickstart

### 1. Generate a smoke dataset

This synthetic dataset is only for pipeline validation. It is not a real AI-art benchmark.

```bash
python scripts/generate_smoke_dataset.py
```

### 2. Prepare the dataset manifest

```bash
python -m ai_art_detector.cli prepare-data --config configs/experiment_smoke.yaml
```

### 3. Train the smoke model

```bash
python -m ai_art_detector.cli train --config configs/experiment_smoke.yaml
```

### 4. Evaluate the checkpoint

```bash
python -m ai_art_detector.cli evaluate --config configs/experiment_smoke.yaml --checkpoint artifacts/training/<run>/checkpoints/best.pt
```

### 5. Run single-image inference

```bash
python -m ai_art_detector.cli predict --config configs/experiment_smoke.yaml --checkpoint artifacts/training/<run>/checkpoints/best.pt --image path/to/image.png
```

## Real Dataset Workflow

### Prepare data

1. Place images in `data/raw/<source>/<label>/...`
2. Run:

```bash
python -m ai_art_detector.cli prepare-data --config configs/experiment.yaml
```

### Train the baseline

```bash
python -m ai_art_detector.cli train --config configs/experiment.yaml
```

### Train the improved experiment

```bash
python -m ai_art_detector.cli train --config configs/experiment_improved.yaml
```

### Evaluate a checkpoint

```bash
python -m ai_art_detector.cli evaluate --config configs/experiment.yaml --checkpoint artifacts/training/<baseline_run>/checkpoints/best.pt
python -m ai_art_detector.cli evaluate --config configs/experiment_improved.yaml --checkpoint artifacts/training/<improved_run>/checkpoints/best.pt
```

### Compare evaluation runs

```bash
python -m ai_art_detector.cli compare-runs \
  --metrics artifacts/evaluation/<baseline_eval>/metrics.json artifacts/evaluation/<improved_eval>/metrics.json \
  --output artifacts/comparison/experiment_comparison.json
```

This writes:
- `artifacts/comparison/experiment_comparison.json`
- `artifacts/comparison/experiment_comparison.md`

## Training Details

### Model support
- `resnet18`
- `efficientnet_b0`
- `tiny_cnn` for smoke checks

### Training behavior
- automatic CPU / CUDA selection
- deterministic seed handling where practical
- mixed precision on CUDA when enabled
- best checkpoint and final checkpoint saving
- validation-based model selection
- configurable optimizer and scheduler

### Training artifacts

Each training run writes to:

```text
artifacts/training/<timestamp>_<experiment_name>/
|-- resolved_config.yaml
|-- history.json
|-- history.csv
|-- training_curve.png
|-- training_summary.json
`-- checkpoints/
    |-- best.pt
    `-- final.pt
```

## Evaluation Details

The evaluation stage can use a validation split for:
- threshold tuning
- temperature scaling

The final reported metrics are then computed on the requested evaluation split, typically `test`.

### Evaluation artifacts

```text
artifacts/evaluation/<timestamp>_<experiment_name>/
|-- resolved_config.yaml
|-- metrics.json
|-- predictions.csv
|-- top_errors.csv
|-- confusion_matrix.png
|-- roc_curve.png
|-- probability_histogram.png
|-- reliability_diagram.png
`-- evaluation_summary.json
```

### Error analysis

`top_errors.csv` records the most confident mistakes so you can inspect:
- false positives on human-made art
- false negatives on AI-generated art
- source clusters that systematically fail
- confidence values that indicate poor calibration

## Inference

### CLI

```bash
python -m ai_art_detector.cli predict \
  --config configs/experiment.yaml \
  --checkpoint artifacts/training/<run>/checkpoints/best.pt \
  --metrics-path artifacts/evaluation/<eval_run>/metrics.json \
  --image path/to/image.png
```

### Output format

Example response shape:

```json
{
  "backend": "torch",
  "calibrated": true,
  "confidence": 0.912,
  "decision": "ai",
  "model_name": "resnet18",
  "predicted_label": "ai",
  "probabilities": {
    "ai": 0.912,
    "human": 0.088
  },
  "probability_ai": 0.912,
  "threshold": 0.47
}
```

## FastAPI API

### Run locally

```bash
set AIAD_CONFIG_PATH=configs/experiment.yaml
set AIAD_MODEL_PATH=artifacts/training/<run>/checkpoints/best.pt
set AIAD_METRICS_PATH=artifacts/evaluation/<eval_run>/metrics.json
python scripts/run_api.py
```

Or with Uvicorn directly:

```bash
uvicorn ai_art_detector.api.app:app --host 0.0.0.0 --port 8000
```

### Endpoints

- `GET /health`
- `POST /predict`
- `POST /predict-batch`

### Example request

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@example.png"
```

## Streamlit Demo

Run:

```bash
python scripts/run_demo.py
```

The demo lets a user:
- upload an image
- view the predicted class
- inspect confidence and class probabilities
- read a short interpretation note about what the score means

The sidebar accepts:
- config path
- checkpoint path
- optional metrics path
- optional ONNX path
- device override
- threshold override

## ONNX Export

```bash
python -m ai_art_detector.cli export-onnx \
  --config configs/experiment.yaml \
  --checkpoint artifacts/training/<run>/checkpoints/best.pt \
  --output artifacts/onnx/model.onnx
```

This writes:
- `artifacts/onnx/model.onnx`
- `artifacts/onnx/model.metadata.json`

## Docker

### API container

```bash
docker build -t ai-art-detector-api .
docker run --rm -p 8000:8000 --env-file .env ai-art-detector-api
```

### Demo container

```bash
docker build -f docker/Dockerfile.demo -t ai-art-detector-demo .
docker run --rm -p 8501:8501 --env-file .env ai-art-detector-demo
```

### Compose

```bash
copy .env.example .env
docker compose up --build
```

## Results and Comparison Strategy

This repo does not ship benchmark numbers because it does not redistribute a public AI-art dataset, and reporting hard-coded metrics without the exact dataset would be misleading.

Instead, the repo ships:
- a baseline experiment config
- an improved experiment config
- a comparison command that produces a Markdown summary
- artifacts needed to discuss calibration, threshold choice, and errors rather than accuracy alone

A strong portfolio workflow is:
1. Run the baseline on your chosen dataset.
2. Run the improved config.
3. Compare ROC-AUC, F1, ECE, and the top-error report.
4. Document where the improved model helps and where it still fails.

## Testing

Run:

```bash
pytest
```

Current tests cover:
- config composition
- dataset preparation
- metric helpers
- experiment comparison writing
- FastAPI behavior with a dummy predictor

## Common Commands

```bash
make smoke-data
make prepare-data CONFIG=configs/experiment.yaml
make train CONFIG=configs/experiment.yaml
make evaluate CONFIG=configs/experiment.yaml CHECKPOINT=artifacts/training/<run>/checkpoints/best.pt
make predict CONFIG=configs/experiment.yaml CHECKPOINT=artifacts/training/<run>/checkpoints/best.pt IMAGE=path/to/image.png
make api
make demo
make test
```

## Limitations

This project is intentionally honest about its limits.

### Dataset leakage risk
- images from the same source can share compression, post-processing, or watermarks
- train/test splits are not a guarantee of true generator isolation
- source-aware evaluation is still necessary

### Domain shift
- a model trained on stylized art may fail on photographs or mixed-media imagery
- a detector tuned on older generators may not generalize to newer ones
- edited images and screenshots can confuse the classifier

### Confidence misuse
- calibrated probabilities are still model estimates, not provenance facts
- high confidence can be wrong under domain shift
- decision thresholds should be selected for the intended use case

### Product limitations
- provenance detection is not cryptographic attribution
- the demo is for model interpretation, not forensic certification

## Future Work

- add grouped source-aware cross-validation
- add richer per-source breakdown reports
- support public benchmark adapters directly
- integrate embedding-based error retrieval for failure analysis
- add model cards and dataset cards as first-class artifacts
- explore CLIP-style or self-supervised feature backbones
- add batch evaluation dashboards

## Status

The repository is now structured as a full end-to-end project. It still depends on an external real dataset for meaningful benchmark numbers, but the code paths for preparation, training, evaluation, inference, API serving, demo usage, comparison, and Dockerization are in place.
