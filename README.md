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

```
knn-digital-recognizer/
├── data/
│   ├── mnist_train.csv       # your training CSV (label + 784 pixels)
│   └── mnist_test.csv        # your test CSV
├── src/
│   ├── __init__.py
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
```

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
```

---

## 🏋️‍♂️ Training & Evaluation

### Train the model

```bash
python src/train.py \
  --data data/mnist_train.csv \
  --k 5 \
  --out knn_model.pkl
```

### Evaluate the model

```bash
python src/evaluate.py \
  --model knn_model.pkl \
  --data data/mnist_test.csv
```

---

## 🧪 Testing & CI

```bash
pytest --maxfail=1 --disable-warnings -q --cov=src/
```

Tests are automatically run on each push/PR via GitHub Actions.

---

## 📖 Quickstart Example

```python
from src.data_loader import load_csv
from src.knn import KNNClassifier

# Load data
X_train, y_train = load_csv("data/mnist_train.csv")
X_test,  y_test  = load_csv("data/mnist_test.csv")

# Initialize, train, and predict
model = KNNClassifier(k=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test[:20])
print("First 20 predictions:", predictions)
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/<name>`)
3. Implement your changes and add tests
4. Update documentation if needed
5. Open a pull request

Please refer to `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

Maintained by Ihor Romaniuk · © 2025
