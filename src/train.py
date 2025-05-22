import argparse
import pickle
from src.data_loader import load_csv
from src.knn import KNNClassifier

def main():
    parser = argparse.ArgumentParser(description="Train KNN on MNIST CSV")
    parser.add_argument("--data", required=True, help="Path to mnist_train.csv")
    parser.add_argument("--k",    type=int, default=3, help="Number of neighbors")
    parser.add_argument("--out",  default="knn_model.pkl", help="Output model path")
    args = parser.parse_args()

    print("[+] Loading data...")
    X, y = load_csv(args.data)

    print(f"[+] Training KNN (k={args.k})...")
    model = KNNClassifier(k=args.k)
    model.fit(X, y)

    print(f"[+] Saving model to {args.out}")
    with open(args.out, "wb") as f:
        pickle.dump(model, f)

    print("[+] Done.")

if __name__ == "__main__":
    main()