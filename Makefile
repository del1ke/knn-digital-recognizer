.PHONY: install test train eval
install:
    pip install -r requirements.txt

test:
    pytest

train:
    python src/train.py …

eval:
    python src/evaluate.py …