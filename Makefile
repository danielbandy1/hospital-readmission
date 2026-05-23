.PHONY: install data train test serve

install:
	pip install -r requirements.txt

data:
	kaggle datasets download -d brandao/diabetes --unzip -p data/

train:
	python3 train_pipeline.py

test:
	python3 -m pytest tests/ -v

serve:
	uvicorn api.serve:app --host 0.0.0.0 --port 8000
