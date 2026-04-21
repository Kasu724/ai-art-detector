PYTHON ?= python
CONFIG ?= configs/experiment.yaml
CHECKPOINT ?=
IMAGE ?=

.PHONY: prepare-data train evaluate predict export-onnx api demo test smoke-data

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
