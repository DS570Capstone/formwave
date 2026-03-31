#!/usr/bin/env python3
"""Train a classifier on features.csv and save a model.

Outputs a saved model at data/metadata/model.pkl and prints evaluation.
"""
from pathlib import Path
import argparse
try:
    import joblib as _joblib
except Exception:
    import pickle as _pickle

    class _JoblibFallback:
        @staticmethod
        def dump(obj, path):
            with open(path, "wb") as f:
                _pickle.dump(obj, f, protocol=4)

    _joblib = _JoblibFallback()
import pandas as pd
import numpy as np
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False


def safe_train(X, y, out_path: Path, random_state=42):
    n = len(y)
    if n < 2:
        print("Not enough samples to train (need >=2).")
        return False

    # choose test size conservatively for small datasets
    if n < 5:
        test_size = 0.5
    else:
        test_size = 0.2

    # attempt stratified split when possible
    stratify = y if len(np.unique(y)) > 1 and min(np.bincount(y)) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # save model
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _joblib.dump(clf, out_path)
    print(f"Saved model to {out_path}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="./data/metadata/features.csv")
    p.add_argument("--out", default="./data/metadata/model.pkl")
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()

    feat_path = Path(args.features)
    if not feat_path.exists():
        print("Features file not found:", feat_path)
        return

    df = pd.read_csv(feat_path)

    required_cols = ["mean_velocity", "std_velocity", "rep_count", "amplitude", "label"]
    if not all(c in df.columns for c in required_cols):
        print("Missing required columns in features CSV. Found:", df.columns.tolist())
        return

    # drop rows with missing labels
    df = df.dropna(subset=["label"]).copy()
    if df.empty:
        print("No labeled rows found in features CSV.")
        return

    # map labels: g->1, b->0; ignore others
    label_map = {"g": 1, "b": 0}
    df["label_map"] = df["label"].map(label_map)
    df = df.dropna(subset=["label_map"]).copy()
    if df.empty:
        print("No rows with recognizable labels (g/b).")
        return

    X = df[["mean_velocity", "std_velocity", "rep_count", "amplitude"]].values
    y = df["label_map"].astype(int).values

    out_path = Path(args.out)
    if not _SKLEARN_AVAILABLE:
        print("scikit-learn not available in this environment.")
        print("Install dependencies, e.g.: pip install scikit-learn pandas numpy scipy joblib")
        return
    safe_train(X, y, out_path, random_state=args.random_state)


if __name__ == "__main__":
    main()
