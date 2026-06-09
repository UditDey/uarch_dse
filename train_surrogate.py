#!/usr/bin/env python3
"""
train_surrogate.py — Train XGBoost surrogate models for IPC and energy.

Usage:
    python3 train_surrogate.py results/dse_data.csv
    python3 train_surrogate.py results/dse_data.csv --model-dir models/
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb

PARAM_COLS = [
    "fetch_width", "decode_width", "issue_width", "commit_width", "dispatch_width",
    "rob_entries", "lq_entries", "sq_entries",
    "l1i_size", "l1d_size", "l1i_assoc", "l1d_assoc",
    "l2_size", "l2_assoc", "bp_type",
]

TARGETS = ["ipc", "power"]


def encode(df):
    """Encode string columns to numeric."""
    df = df.copy()
    size_map = {"16kB": 16384, "32kB": 32768, "64kB": 65536,
                "128kB": 131072, "256kB": 262144, "512kB": 524288, "1MB": 1048576}
    for col in ["l1i_size", "l1d_size", "l2_size"]:
        df[col] = df[col].replace(size_map)
    df["bp_type"] = df["bp_type"].replace(
        {"LocalBP": 0, "BiModeBP": 1, "TournamentBP": 2, "TAGE": 3})
    return df


def split(n, test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    split = int(n * (1 - test_frac))
    return idx[:split], idx[split:]


def metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return r2, mae, mape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="results/dse_data.csv")
    ap.add_argument("--model-dir", default="models")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = encode(df)
    df = df.dropna(subset=TARGETS)
    print(f"Loaded {len(df)} samples from {args.csv}")

    X = df[PARAM_COLS].values.astype(float)
    train_idx, test_idx = split(len(df))
    X_train, X_test = X[train_idx], X[test_idx]

    print(f"Train: {len(train_idx)}  Test: {len(test_idx)}\n")

    os.makedirs(args.model_dir, exist_ok=True)

    for target in TARGETS:
        y = df[target].values
        y_train, y_test = y[train_idx], y[test_idx]

        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        r2, mae, mape = metrics(y_test, y_pred)

        print(f"{target}:")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  MAPE: {mape:.2f}%")

        # Feature importance
        imp = model.feature_importances_
        top = np.argsort(imp)[-5:][::-1]
        print(f"  Top features: {', '.join(f'{PARAM_COLS[i]}={imp[i]:.3f}' for i in top)}")

        # Worst predictions
        errors = np.abs(y_test - y_pred)
        worst = np.argsort(errors)[-3:]
        print(f"  Worst 3:")
        for i in worst:
            print(f"    actual={y_test[i]:.4f}  pred={y_pred[i]:.4f}")

        # Save
        path = os.path.join(args.model_dir, f"{target}.json")
        model.save_model(path)
        print(f"  Saved: {path}\n")

    # Dataset summary
    print("Dataset summary:")
    for target in TARGETS:
        v = df[target]
        print(f"  {target}: min={v.min():.4f}  max={v.max():.4f}  "
              f"mean={v.mean():.4f}  std={v.std():.4f}")


if __name__ == "__main__":
    main()
