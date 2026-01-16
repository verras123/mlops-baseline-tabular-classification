# -*- coding: utf-8 -*-
"""
Reproducible ML pipeline (binary classification) using a CSV dataset.

Outputs:
- confusion matrix PNG
- ROC curve PNG
- trained model joblib
- report.json with CV and test metrics
"""

import os
import json
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def infer_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    import inspect

    ohe_kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        ohe_kwargs["sparse_output"] = False
    else:
        ohe_kwargs["sparse"] = False

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(**ohe_kwargs)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def evaluate_model(
    name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    out_dir: str,
    cv_splits: int = 5,
    seed: int = 42,
) -> dict:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    scoring = {
        "acc": "accuracy",
        "prec": "precision",
        "rec": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv_res = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        metrics["test_roc_auc"] = float(roc_auc_score(y_test, y_proba))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_confusion_matrix.png"), dpi=160)
    plt.close()

    if y_proba is not None:
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title(f"ROC Curve - {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_roc_curve.png"), dpi=160)
        plt.close()

    joblib.dump(model, os.path.join(out_dir, f"{name}_model.joblib"))

    summary = {
        "name": name,
        "cv": {k: [float(x) for x in cv_res[f"test_{k}"]] for k in scoring.keys()},
        "cv_mean": {k: float(np.mean(cv_res[f"test_{k}"])) for k in scoring.keys()},
        "cv_std": {k: float(np.std(cv_res[f"test_{k}"])) for k in scoring.keys()},
        "test_metrics": metrics,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/dataset_grande.csv", help="Path to CSV dataset")
    parser.add_argument("--target", type=str, default="target", help="Target column name")
    parser.add_argument("--out", type=str, default="outputs", help="Output folder")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cv_splits", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    ensure_dir(args.out)

    if not os.path.exists(args.csv):
        raise FileNotFoundError(
            f"CSV not found: {args.csv}\n"
            "Generate it first with src/gerar_csv_grande.py or pass --csv with the correct path."
        )

    df = pd.read_csv(args.csv)

    if args.target not in df.columns:
        raise ValueError(
            f"Target column '{args.target}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(axis=0, how="all").reset_index(drop=True)

    X = df.drop(columns=[args.target])
    y = df[args.target].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    numeric_cols, categorical_cols = infer_feature_types(df, args.target)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    lr = LogisticRegression(max_iter=2000)
    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=args.seed,
        class_weight="balanced",
        n_jobs=-1,
    )

    lr_pipe = Pipeline(steps=[("prep", preprocessor), ("clf", lr)])
    rf_pipe = Pipeline(steps=[("prep", preprocessor), ("clf", rf)])

    results = []
    print("[INFO] Training/evaluating Logistic Regression...")
    results.append(
        evaluate_model(
            name="logreg",
            model=lr_pipe,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            out_dir=args.out,
            cv_splits=args.cv_splits,
            seed=args.seed,
        )
    )

    print("[INFO] Training/evaluating Random Forest...")
    results.append(
        evaluate_model(
            name="random_forest",
            model=rf_pipe,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            out_dir=args.out,
            cv_splits=args.cv_splits,
            seed=args.seed,
        )
    )

    def pick_score(res: dict) -> float:
        return float(res["cv_mean"].get("roc_auc", res["cv_mean"].get("f1", 0.0)))

    best = sorted(results, key=pick_score, reverse=True)[0]

    report = {
        "dataset_path": os.path.abspath(args.csv),
        "target": args.target,
        "n_rows": int(df.shape[0]),
        "n_features_raw": int(X.shape[1]),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "test_size": float(args.test_size),
        "cv_splits": int(args.cv_splits),
        "seed": int(args.seed),
        "results": results,
        "best_model": best["name"],
    }

    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n========== SUMMARY ==========")
    for r in results:
        print(f"\nModel: {r['name']}")
        print(
            f"CV mean: acc={r['cv_mean']['acc']:.4f}  prec={r['cv_mean']['prec']:.4f}  "
            f"rec={r['cv_mean']['rec']:.4f}  f1={r['cv_mean']['f1']:.4f}  "
            f"roc_auc={r['cv_mean']['roc_auc']:.4f}"
        )
        tm = r["test_metrics"]
        print(
            f"TEST: acc={tm['test_accuracy']:.4f} prec={tm['test_precision']:.4f} "
            f"rec={tm['test_recall']:.4f} f1={tm['test_f1']:.4f} "
            + (f"roc_auc={tm['test_roc_auc']:.4f}" if "test_roc_auc" in tm else "")
        )

    print(f"\n[INFO] Best model by CV score: {report['best_model']}")
    print(f"[INFO] Outputs saved in: {os.path.abspath(args.out)}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
