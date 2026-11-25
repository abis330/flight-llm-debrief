.PHONY: install test data dataset baselines train eval serve
install:
	pip install -e ".[train,anthropic,dev]"
test:
	pytest -q
data:
	flightlake simulate --n 1000 --out data/raw --workers 4 && flightlake ingest data/raw --lake data/lake --workers 4
dataset:
	flightdebrief build-dataset --lake data/lake --out data/sft
baselines:
	flightdebrief baselines --data data/sft --out results
train:
	python -m flightdebrief.train_sft --config configs/qwen2.5-1.5b-lora.yaml
eval:
	flightdebrief eval-model --backend hf --model Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/qwen2.5-1.5b-debrief/final_adapter --name finetuned
serve:
	docker compose up --build
