.PHONY: run install activate

activate:
	conda activate drone_vision

install:
	pip install -r requirements.txt

run:
	python atello.py
