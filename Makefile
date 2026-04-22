PYTHON ?= python
CONFIG ?= configs/experiment.yaml
CHECKPOINT ?=
IMAGE ?=
OUTPUT_DIR ?= data/raw/hf_art_images_ai_and_real
HUMAN_LIMIT ?= 3000
AI_LIMIT ?= 3000

.PHONY: download-real-data download-anime-data prepare-data train evaluate predict export-onnx api demo test smoke-data

download-real-data:
	$(PYTHON) -m ai_art_detector.cli download-real-dataset --output-dir $(OUTPUT_DIR)

download-anime-data:
	$(PYTHON) -m ai_art_detector.cli download-anime-dataset --output-dir data/raw/anime_social_filter --human-limit $(HUMAN_LIMIT) --ai-limit $(AI_LIMIT)

prepare-data:
	$(PYTHON) -m ai_art_detector.cli prepare-data --config $(CONFIG)

train:
	$(PYTHON) -m ai_art_detector.cli train --config $(CONFIG)

evaluate:
	$(PYTHON) -m ai_art_detector.cli evaluate --config $(CONFIG) --checkpoint $(CHECKPOINT)

predict:
	$(PYTHON) -m ai_art_detector.cli predict --config $(CONFIG) --checkpoint $(CHECKPOINT) --image $(IMAGE)

export-onnx:
	$(PYTHON) -m ai_art_detector.cli export-onnx --config $(CONFIG) --checkpoint $(CHECKPOINT) --output artifacts/model.onnx

api:
	$(PYTHON) scripts/run_api.py

demo:
	$(PYTHON) scripts/run_demo.py

smoke-data:
	$(PYTHON) scripts/generate_smoke_dataset.py

test:
	pytest
