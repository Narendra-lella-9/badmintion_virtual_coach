from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train point activity classifier from frame-level feature CSVs."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Numpy .npz file with X (n,features) and y (n,) arrays",
    )
    parser.add_argument("--output-model", required=True, type=Path)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=22)
    parser.add_argument("--min-samples-leaf", type=int, default=4)
    parser.add_argument(
        "--optimize-threshold",
        action="store_true",
        help="Search probability threshold for best rally F1 on validation split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.dataset)
    X = data["X"]
    y = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds, digits=4))

    if args.optimize_threshold and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        best_thr = 0.5
        best_f1 = -1.0
        for thr in np.arange(0.30, 0.91, 0.02):
            pred_thr = (proba >= thr).astype(np.int32)
            tp = int(np.sum((pred_thr == 1) & (y_test == 1)))
            fp = int(np.sum((pred_thr == 1) & (y_test == 0)))
            fn = int(np.sum((pred_thr == 0) & (y_test == 1)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)
        print(f"Suggested threshold for rally detection: {best_thr:.2f} (F1={best_f1:.4f})")

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_model)
    print(f"Saved model: {args.output_model}")
    print(f"Feature dimension used: {X.shape[1]}")


if __name__ == "__main__":
    main()
