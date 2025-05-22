# KNN Digital Recognizer

A pure-NumPy implementation of K-Nearest Neighbors for handwritten-digit recognition (MNIST).  
Includes CSV data loading, pixel normalization, training, evaluation, unit tests and GitHub Actions CI.

---

## 🚀 Overview

This project shows how to:

- Load MNIST-style CSVs with `pandas` & `NumPy`
- Normalize raw pixel features
- Implement Euclidean-distance + majority-vote KNN from scratch
- Train & serialize a model (`pickle`)
- Evaluate performance (accuracy + confusion matrix)
- Structure a modern Python repo with CLI, tests, and CI

---

## 🔍 Repo Layout

knn-digital-recognizer/
├── data/
│   ├── mnist_train.csv       # your training CSV (label + 784 pixels)
│   └── mnist_test.csv        # your test CSV
├── src/
│   ├── init.py
│   ├── data_loader.py        # load_csv()
│   ├── feature_extractor.py  # normalize_pixels()
│   ├── knn.py                # KNNClassifier()
│   ├── train.py              # train & save model
│   └── evaluate.py           # load model & report metrics
├── tests/
│   └── test_knn.py           # unit-tests for distance & voting logic
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
---

## ⚙️ Installation

```bash
# 1. Clone & enter
git clone https://github.com/del1ke/knn-digital-recognizer.git
cd knn-digital-recognizer

# 2. Create & activate venv
python3 -m venv .venv
source .venv/bin/activate

# 3. Install deps
pip install -r requirements.txt

# 4. Download data
#    Place your `mnist_train.csv` & `mnist_test.csv` into `data/`

## 🏋️‍♂️ Training & Evaluation

# Train a KNN (k=5) and save to knn_model.pkl
python src/train.py \
  --data data/mnist_train.csv \
  --k 5 \
  --out knn_model.pkl

# Evaluate on the test set
python src/evaluate.py \
  --model knn_model.pkl \
  --data data/mnist_test.csv

## 🧪 Testing & CI

pytest --maxfail=1 --disable-warnings -q --cov=src/

## 📖 Quickstart Example

from src.data_loader import load_csv
from src.knn import KNNClassifier

X_train, y_train = load_csv("data/mnist_train.csv")
X_test,  y_test  = load_csv("data/mnist_test.csv")

model = KNNClassifier(k=3)
model.fit(X_train, y_train)
preds = model.predict(X_test[:20])
print("First 20 predictions:", preds)

## 📜 License

Licensed under the [MIT License](LICENSE).

---

<sup>Maintained by Ihor Romaniuk (https://github.com/del1ke) · © 2025</sup>