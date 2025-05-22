import argparse
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
from src.data_loader import load_csv

def main():
    parser = argparse.ArgumentParser(description="Evaluate KNN model")
    parser.add_argument("--model", required=True, help="Path to knn_model.pkl")
    parser.add_argument("--data",  required=True, help="Path to mnist_test.csv")
    args = parser.parse_args()

    print("[+] Loading test data...")
    X_test, y_test = load_csv(args.data)

    print(f"[+] Loading model from {args.model}")
    model = pickle.load(open(args.model, "rb"))

    print("[+] Predicting...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()