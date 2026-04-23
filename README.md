# AI Art Detector

An image classifier for detecting whether an artwork image is more likely AI-generated or human-made.

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
- the README documents domain-shift and leakage concerns

## In this Repo

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
- dark-only Streamlit demo
- single-image CLI inference
- ONNX export path

### Engineering / Reproducibility
- config-driven experiments under `configs/`
- run artifacts grouped by stage under `artifacts/`
- tests for config loading, dataset preparation, metrics, and API behavior
- Dockerfile for the API and an additional demo Dockerfile
- `Makefile` and `scripts/` wrappers for common tasks
- smoke dataset generator for pipeline checks

## Experiment Lineup

The repository ships baseline, comparison, and smoke configs:

| Experiment | Config | Purpose | Backbone |
| --- | --- | --- | --- |
| Generic baseline | `configs/experiment.yaml` | Template transfer-learning baseline for your own folder dataset | `resnet18` |
| Generic comparison | `configs/experiment_improved.yaml` | Template comparison run with heavier augmentation | `efficientnet_b0` |
| Real baseline | `configs/experiment_real_art_hf.yaml` | Baseline on the public Hugging Face art dataset | `resnet18` |
| Real comparison | `configs/experiment_real_art_hf_improved.yaml` | Same-dataset comparison run with stronger augmentation | `efficientnet_b0` |
| Anime moderation | `configs/experiment_anime_social_transfer.yaml` | Warm-started fine-tune for anime/cartoon-style moderation | `efficientnet_b0` |
| Fanart moderation v2 | `configs/experiment_anime_fanart_v2_transfer.yaml` | Second-stage fine-tune for polished human fanart vs AI anime | `efficientnet_b0` |
| Fanart moderation v3 | `configs/experiment_anime_fanart_v3_transfer.yaml` | Broader fanart moderation fine-tune with a wider AI-side source mix | `efficientnet_b0` |
| Fanart moderation v4 | `configs/experiment_anime_fanart_v4_transfer.yaml` | AI-recall-focused fine-tune for stricter moderation | `efficientnet_b0` |
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

### 1. Download the real public dataset

```bash
python -m ai_art_detector.cli download-real-dataset --output-dir data/raw/hf_art_images_ai_and_real
```

### 2. Prepare the real dataset manifest

```bash
python -m ai_art_detector.cli prepare-data --config configs/experiment_real_art_hf.yaml
```

### 3. Train the real baseline

```bash
python -m ai_art_detector.cli train --config configs/experiment_real_art_hf.yaml
```

### 4. Evaluate the real baseline

```bash
python -m ai_art_detector.cli evaluate --config configs/experiment_real_art_hf.yaml --checkpoint artifacts/training/<run>/checkpoints/best.pt
```

### 5. Train the real comparison variant

```bash
python -m ai_art_detector.cli train --config configs/experiment_real_art_hf_improved.yaml
```

### 6. Evaluate the comparison variant

```bash
python -m ai_art_detector.cli evaluate --config configs/experiment_real_art_hf_improved.yaml --checkpoint artifacts/training/<run>/checkpoints/best.pt
```

### 7. Run single-image inference

```bash
python -m ai_art_detector.cli predict --config configs/experiment_real_art_hf_improved.yaml --checkpoint artifacts/training/<run>/checkpoints/best.pt --image path/to/image.png
```

### Optional smoke-only pipeline check

This synthetic dataset is only for pipeline validation. It is not a real AI-art benchmark.

```bash
python scripts/generate_smoke_dataset.py
python -m ai_art_detector.cli prepare-data --config configs/experiment_smoke.yaml
python -m ai_art_detector.cli train --config configs/experiment_smoke.yaml
```

## Real Dataset Workflow

The repo includes a built-in adapter for the public `DataScienceProject/Art_Images_Ai_And_Real_` Hugging Face dataset.

### Download and materialize data

```bash
python scripts/download_real_dataset.py --output-dir data/raw/hf_art_images_ai_and_real
```

The downloader writes:
- `data/raw/hf_art_images_ai_and_real/ai/*.png`
- `data/raw/hf_art_images_ai_and_real/human/*.png`
- `data/raw/hf_art_images_ai_and_real_download_summary.json`

### Prepare data

Run:

```bash
python -m ai_art_detector.cli prepare-data --config configs/experiment_real_art_hf.yaml
```

### Train the baseline

```bash
python -m ai_art_detector.cli train --config configs/experiment_real_art_hf.yaml
```

### Train the comparison experiment

```bash
python -m ai_art_detector.cli train --config configs/experiment_real_art_hf_improved.yaml
```

### Evaluate a checkpoint

```bash
python -m ai_art_detector.cli evaluate --config configs/experiment_real_art_hf.yaml --checkpoint artifacts/training/<baseline_run>/checkpoints/best.pt
python -m ai_art_detector.cli evaluate --config configs/experiment_real_art_hf_improved.yaml --checkpoint artifacts/training/<improved_run>/checkpoints/best.pt
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

## Anime Moderation Workflow

This repo also includes a style-focused workflow aimed at anime/cartoon-heavy moderation scenarios such as art-sharing platforms.

### Built-in sources

- Human-style anime / illustration:
  - `sayurio/anime-art-image`
  - `Dhiraj45/Animes`
- AI-style anime:
  - `ShoukanLabs/OpenNiji-0_32237`
  - `ShoukanLabs/OpenNiji-65001_100000`

### Download the anime moderation dataset

```bash
python -m ai_art_detector.cli download-anime-dataset --output-dir data/raw/anime_social_filter --human-limit 3000 --ai-limit 3000
```

This materializes a balanced 2-class folder tree with per-source subdirectories so preparation can stratify by both label and source.

### Prepare the anime moderation manifest

```bash
python -m ai_art_detector.cli prepare-data --config configs/experiment_anime_social_transfer.yaml
```

### Train the anime moderation model

```bash
python -m ai_art_detector.cli train --config configs/experiment_anime_social_transfer.yaml
```

### Evaluate the anime moderation model

```bash
python -m ai_art_detector.cli evaluate --config configs/experiment_anime_social_transfer.yaml --checkpoint artifacts/training/<run>/checkpoints/best.pt
```

## Fanart Moderation Workflow

This workflow is a tighter proxy for polished Twitter-style anime art and fanart.

### Built-in human curation logic

Human examples are streamed from `ShinoharaHare/Danbooru-2024-Filtered-1M` and filtered to keep:

- ratings `g` or `s`
- artist-tagged posts
- posts with character or copyright tags
- safe score >= `0.8`
- polished score >= `0.85`
- aesthetic score >= `5.0`
- no `ai-generated` or `ai-assisted` tag

### Download the curated fanart dataset

```bash
python -m ai_art_detector.cli download-anime-fanart-dataset --output-dir data/raw/anime_fanart_filter_v2 --human-limit 3000 --ai-limit 3000
```

### Prepare, train, and evaluate

```bash
python -m ai_art_detector.cli prepare-data --config configs/experiment_anime_fanart_v2_transfer.yaml
python -m ai_art_detector.cli train --config configs/experiment_anime_fanart_v2_transfer.yaml
python -m ai_art_detector.cli evaluate --config configs/experiment_anime_fanart_v2_transfer.yaml --checkpoint artifacts/training/<run>/checkpoints/best.pt
```

### Download the broader fanart v3 dataset

This variant keeps the same curated human fanart source and broadens the AI side
from two OpenNiji ranges to three:

- `ShoukanLabs/OpenNiji-0_32237`
- `ShoukanLabs/OpenNiji-32238_65000`
- `ShoukanLabs/OpenNiji-65001_100000`

```bash
python -m ai_art_detector.cli download-anime-fanart-v3-dataset --output-dir data/raw/anime_fanart_filter_v3 --human-limit 3000 --ai-limit 3000
python -m ai_art_detector.cli prepare-data --config configs/experiment_anime_fanart_v3_transfer.yaml
python -m ai_art_detector.cli train --config configs/experiment_anime_fanart_v3_transfer.yaml
python -m ai_art_detector.cli evaluate --config configs/experiment_anime_fanart_v3_transfer.yaml --checkpoint artifacts/training/<run>/checkpoints/best.pt
```

### Download the AI-recall-focused fanart v4 dataset

This variant is intended for stricter moderation. It keeps the curated human
fanart source, adds a real Ghibli-style human source to avoid treating the
whole style as AI, and adds auxiliary AI-generated Ghibli-style sources.

```bash
python -m ai_art_detector.cli download-anime-fanart-v4-dataset --output-dir data/raw/anime_fanart_filter_v4 --human-limit 4000 --ai-limit 4200
python -m ai_art_detector.cli prepare-data --config configs/experiment_anime_fanart_v4_transfer.yaml
python -m ai_art_detector.cli train --config configs/experiment_anime_fanart_v4_transfer.yaml
python -m ai_art_detector.cli evaluate --config configs/experiment_anime_fanart_v4_transfer.yaml --checkpoint artifacts/training/<run>/checkpoints/best.pt
```

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
  --config configs/experiment_real_art_hf_improved.yaml \
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
set AIAD_CONFIG_PATH=configs/experiment_anime_fanart_v4_transfer.yaml
set AIAD_MODEL_PATH=artifacts/training/20260422_223526_anime_fanart_v4_recall_efficientnet_b0/checkpoints/best.pt
set AIAD_METRICS_PATH=artifacts/evaluation/20260423_000248_anime_fanart_v4_recall_efficientnet_b0/metrics.json
set AIAD_THRESHOLD=0.4
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

The demo is intentionally dark-only and minimal. It does not expose a light-mode
or theme switch, and no `AIAD_DEMO_THEME` environment variable is required.

It uses the same model-related environment variables as the API when they are
present in `.env`:

- `AIAD_CONFIG_PATH`
- `AIAD_MODEL_PATH`
- `AIAD_METRICS_PATH`
- `AIAD_ONNX_PATH`
- `AIAD_THRESHOLD`
- `AIAD_DEVICE`

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
  --config configs/experiment_real_art_hf_improved.yaml \
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

The repo now includes a real public-dataset workflow and example benchmark numbers from an executed run on April 21, 2026.

### Public dataset used

- Dataset: `DataScienceProject/Art_Images_Ai_And_Real_`
- Materialized samples: 2,839 images
- Class balance after download: 1,420 AI / 1,419 human
- Project split after manifest preparation: 1,987 train / 426 val / 426 test

### Held-out test metrics

| Experiment | Config | Accuracy | Precision | Recall | F1 | ROC-AUC | ECE | Threshold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | `configs/experiment_real_art_hf.yaml` | 0.9507 | 0.9660 | 0.9343 | 0.9499 | 0.9870 | 0.0899 | 0.100 |
| Comparison variant | `configs/experiment_real_art_hf_improved.yaml` | 0.9883 | 0.9906 | 0.9859 | 0.9882 | 0.9972 | 0.0137 | 0.375 |

### Interpretation

- The `efficientnet_b0` comparison run materially improved both discrimination and calibration.
- The best real run made 5 mistakes on the 426-image held-out test split: 2 false positives and 3 false negatives.
- Temperature scaling reduced validation log loss for both runs, and the stronger model also kept test-time ECE low.
- The comparison summary generated by the repo is stored at `artifacts/comparison/real_art_hf_baseline_vs_efficientnet.md`.

### Anime moderation fine-tune

On April 22, 2026, the repo was also used to build a style-focused anime moderation dataset with 6,000 images:

- Human: 3,000
- AI: 3,000
- Sources: `sayurio_anime_art`, `dhiraj45_animes`, `open_niji_0_32237`, `open_niji_65001_100000`
- Split: 4,200 train / 900 val / 900 test

Held-out anime-style test metrics for the fine-tuned checkpoint:

| Experiment | Accuracy | Precision | Recall | F1 | ROC-AUC | ECE | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Generic EfficientNet on anime split | 0.8022 | 0.7491 | 0.9089 | 0.8213 | 0.8742 | 0.2417 | 0.725 |
| Anime-specific fine-tune | 0.9844 | 0.9781 | 0.9911 | 0.9845 | 0.9980 | 0.0079 | 0.300 |

Operational threshold sweep for the anime fine-tune on the held-out test split:

- `0.3`: precision `0.9781`, recall `0.9911`
- `0.5`: precision `0.9889`, recall `0.9889`
- `0.7`: precision `0.9933`, recall `0.9844`

The anime fine-tune reduced the generic model's false positives on stylized art substantially, but the hardest remaining human false positives still cluster in the `sayurio_anime_art` source. That is a good reminder that illustration-heavy human art can still sit very close to modern anime generators in feature space.

### Fanart-focused v2 fine-tune

To better match polished social-media fanart, the repo was then extended with a curated fanart proxy:

- Human source: filtered `ShinoharaHare/Danbooru-2024-Filtered-1M`
- AI sources: `ShoukanLabs/OpenNiji-0_32237`, `ShoukanLabs/OpenNiji-65001_100000`
- Size: 6,000 images
- Split: 4,200 train / 900 val / 900 test

On the held-out fanart-style test split:

| Experiment | Accuracy | Precision | Recall | F1 | ROC-AUC | ECE | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prior anime model on fanart split | 0.7478 | 0.7451 | 0.7533 | 0.7492 | 0.8057 | 0.3518 | 0.850 |
| Fanart-focused v2 model | 0.9400 | 0.9400 | 0.9400 | 0.9400 | 0.9848 | 0.0498 | 0.750 |

The key moderation improvement is false positives on real fanart:

- Prior anime model: `116 / 450` human fanart samples falsely flagged as AI
- Fanart-focused v2 model: `27 / 450`

Suggested operating points on the held-out fanart split:

- `0.5`: precision `0.8966`, recall `0.9822`
- `0.7`: precision `0.9342`, recall `0.9467`
- tuned `0.75`: precision `0.9400`, recall `0.9400`

### Fanart-focused v3 fine-tune

To reduce overfitting to a narrower AI source mix, the repo was then extended
with a third fanart-stage dataset:

- Human source: filtered `ShinoharaHare/Danbooru-2024-Filtered-1M`
- AI sources: `ShoukanLabs/OpenNiji-0_32237`, `ShoukanLabs/OpenNiji-32238_65000`, `ShoukanLabs/OpenNiji-65001_100000`
- Size: 6,000 images
- Split: 4,200 train / 900 val / 900 test

On the held-out v3 fanart-style test split:

| Experiment | Accuracy | Precision | Recall | F1 | ROC-AUC | ECE | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v2 checkpoint on v3 split | 0.9533 | 0.9359 | 0.9733 | 0.9542 | 0.9884 | 0.0570 | 0.700 |
| Fanart-focused v3 model | 0.9622 | 0.9622 | 0.9622 | 0.9622 | 0.9927 | 0.0192 | 0.575 |

The operational tradeoff on the v3 split is more balanced than v2:

- Human false positives dropped from `30 / 450` to `17 / 450`
- AI misses rose from `12 / 450` to `17 / 450`
- Calibration improved materially with ECE moving from `0.0570` to `0.0192`

Suggested operating points on the held-out v3 fanart split:

- `0.4`: precision `0.9464`, recall `0.9800`
- tuned `0.575`: precision `0.9622`, recall `0.9622`
- `0.7`: precision `0.9705`, recall `0.9489`

For a platform workflow where false accusations on real fanart are especially
costly, `v3` is the better default checkpoint. If you want a stricter first-pass
AI catch rate, lower the threshold toward `0.4` or keep `v2` as a more aggressive
recall-oriented screening model.

### Fanart-focused v4 recall fine-tune

After seeing that the balanced v3 model still let too many AI images through,
the repo was extended with an AI-recall-focused v4 run:

- Human sources: filtered `ShinoharaHare/Danbooru-2024-Filtered-1M`, plus real-labeled samples from `pulnip/ghibli-dataset`
- AI sources: three OpenNiji ranges, non-real labels from `pulnip/ghibli-dataset`, and `filberthamijoyo/AI_Generated_Ghibli`
- Size after validation: 8,271 images
- Split: 5,791 train / 1,241 val / 1,239 test
- One truncated image was detected and excluded during preparation

On the held-out v4 fanart-style test split:

| Experiment | Accuracy | Precision | Recall | F1 | ROC-AUC | ECE | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 checkpoint on v4 split | 0.8878 | 0.8463 | 0.9562 | 0.8979 | 0.9655 | 0.0391 | 0.425 |
| Fanart-focused v4 model | 0.9475 | 0.9233 | 0.9797 | 0.9506 | 0.9911 | 0.0561 | 0.575 |

The key moderation improvement is AI recall on the broader split:

- v3 checkpoint: `28 / 639` AI samples missed as human
- v4 checkpoint: `13 / 639` AI samples missed as human

Suggested stricter operating points for v4:

- tuned `0.575`: precision `0.9233`, recall `0.9797`, AI misses `13 / 639`
- `0.4`: precision `0.8799`, recall `0.9859`, AI misses `9 / 639`
- `0.3`: precision `0.8520`, recall `0.9906`, AI misses `6 / 639`

For an art-sharing platform, `AIAD_THRESHOLD=0.4` is a more appropriate first-pass
filter than the balanced threshold. It should be treated as a review/flagging
threshold, not as proof that an image is AI-generated.

The repo also ships:
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
- real dataset download materialization
- dataset preparation
- metric helpers
- experiment comparison writing
- FastAPI behavior with a dummy predictor

## Common Commands

```bash
make download-real-data
make download-anime-data
make download-anime-fanart-data
make download-anime-fanart-v3-data
make download-anime-fanart-v4-data
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
- the included Hugging Face dataset arrives with its own upstream train/test folders, but this project re-prepares a fresh manifest split for reproducible internal experimentation; treat the resulting score as a project benchmark, not the dataset author's canonical leaderboard number

### Domain shift
- a model trained on stylized art may fail on photographs or mixed-media imagery
- a detector tuned on older generators may not generalize to newer ones
- edited images and screenshots can confuse the classifier
- an anime-focused detector can outperform a generic art detector on anime/cartoon imagery while still underperforming on other illustration communities or on styles not represented in the fine-tuning sources

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

The repository is now a project with a tested public-dataset adapter, trained real checkpoints, anime- and fanart-focused transfer runs through `v4`, evaluation artifacts, ONNX export, a working FastAPI service, and a working Streamlit demo.
