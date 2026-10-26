install:
	pip install -r requirements.txt
train:
	PYTHONPATH=src python -m fraud_detection.train
test:
	PYTHONPATH=src pytest -q
serve:
	PYTHONPATH=src uvicorn fraud_detection.api:app --reload
