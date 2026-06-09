#!/usr/bin/env python3
"""
dse_optimize.py — Find Pareto-optimal configs using saved XGBoost surrogates.

Usage:
    python3 dse_optimize.py
    python3 dse_optimize.py --model-dir models/ --n-gen 300
"""

import argparse
import numpy as np
import xgboost as xgb
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# ── Design space ─────────────────────────────────────────────────────

PARAM_NAMES = [
    "fetch_width", "decode_width", "issue_width", "commit_width", "dispatch_width",
    "rob_entries", "lq_entries", "sq_entries",
    "l1i_size", "l1d_size", "l1i_assoc", "l1d_assoc",
    "l2_size", "l2_assoc", "bp_type",
]

PARAM_VALUES = [
    [2, 4, 8],           # fetch_width
    [2, 4, 8],           # decode_width
    [2, 4, 8],           # issue_width
    [2, 4, 8],           # commit_width
    [2, 4, 8],           # dispatch_width
    [32, 64, 128, 192, 256],  # rob_entries
    [16, 32, 64],        # lq_entries
    [16, 32, 64],        # sq_entries
    [16384, 32768, 65536],     # l1i_size
    [16384, 32768, 65536],     # l1d_size
    [2, 4],              # l1i_assoc
    [2, 4],              # l1d_assoc
    [131072, 262144, 524288, 1048576],  # l2_size
    [4, 8, 16],          # l2_assoc
    [0, 1, 2, 3],        # bp_type
]

SIZE_LABELS = {16384: "16kB", 32768: "32kB", 65536: "64kB",
               131072: "128kB", 262144: "256kB", 524288: "512kB", 1048576: "1MB"}
BP_LABELS = {0: "LocalBP", 1: "BiModeBP", 2: "TournamentBP", 3: "TAGE"}


def snap(X):
    """Snap continuous values to nearest discrete values."""
    out = np.zeros_like(X)
    for j, vals in enumerate(PARAM_VALUES):
        arr = np.array(vals)
        for i in range(X.shape[0]):
            out[i, j] = arr[np.argmin(np.abs(arr - X[i, j]))]
    return out


def fix_widths(X):
    """Enforce fetch >= decode >= dispatch >= issue, commit >= issue."""
    for i in range(X.shape[0]):
        w = sorted([X[i, 0], X[i, 1], X[i, 4], X[i, 2], X[i, 3]], reverse=True)
        X[i, 0] = w[0]  # fetch
        X[i, 1] = w[1]  # decode
        X[i, 4] = w[2]  # dispatch
        X[i, 2] = w[3]  # issue
        X[i, 3] = max(w[4], X[i, 2])  # commit >= issue
    return X


class DSEProblem(Problem):
    def __init__(self, ipc_model, power_model):
        xl = np.array([min(v) for v in PARAM_VALUES], dtype=float)
        xu = np.array([max(v) for v in PARAM_VALUES], dtype=float)
        super().__init__(n_var=len(PARAM_NAMES), n_obj=2, xl=xl, xu=xu)
        self.ipc_model = ipc_model
        self.power_model = power_model

    def _evaluate(self, X, out, *args, **kwargs):
        X_d = fix_widths(snap(X))
        ipc = self.ipc_model.predict(X_d)
        power = self.power_model.predict(X_d)
        out["F"] = np.column_stack([-ipc, power])  # minimize -IPC, minimize power


def label(val, name):
    if name in ("l1i_size", "l1d_size", "l2_size"):
        return SIZE_LABELS.get(int(val), str(int(val)))
    if name == "bp_type":
        return BP_LABELS.get(int(val), str(int(val)))
    return str(int(val))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--n-gen", type=int, default=200)
    ap.add_argument("--pop-size", type=int, default=100)
    args = ap.parse_args()

    # Load models
    ipc_model = xgb.XGBRegressor()
    ipc_model.load_model(f"{args.model_dir}/ipc.json")
    power_model = xgb.XGBRegressor()
    power_model.load_model(f"{args.model_dir}/power.json")
    print("Loaded models\n")

    # Optimize
    print(f"Running NSGA-II: pop={args.pop_size} gen={args.n_gen}")
    result = minimize(
        DSEProblem(ipc_model, power_model),
        NSGA2(pop_size=args.pop_size, sampling=FloatRandomSampling(),
              crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20),
              eliminate_duplicates=True),
        termination=get_termination("n_gen", args.n_gen),
        seed=42, verbose=False,
    )

    # Extract and clean Pareto front
    X_opt = fix_widths(snap(result.X))
    F = result.F

    # Deduplicate
    seen = set()
    keep = []
    for i in range(len(X_opt)):
        key = tuple(X_opt[i].astype(int))
        if key not in seen:
            seen.add(key)
            keep.append(i)
    X_opt, F = X_opt[keep], F[keep]

    # Sort by IPC descending
    order = np.argsort(F[:, 0])
    X_opt, F = X_opt[order], F[order]

    # Print
    print(f"\nPareto front: {len(X_opt)} configs\n")
    print(f"{'#':>3s} {'IPC':>7s} {'Power (W)':>14s}   "
          f"{'width':>5s} {'ROB':>4s} {'L1I':>5s} {'L1D':>5s} {'L2':>6s} {'BP':>12s}")
    print("-" * 80)

    for i in range(len(X_opt)):
        c = X_opt[i]
        print(f"{i+1:3d} {-F[i,0]:7.4f} {F[i,1]:14.4f}   "
              f"{int(c[2]):5d} {int(c[5]):4d} "
              f"{label(c[8], 'l1i_size'):>5s} {label(c[9], 'l1d_size'):>5s} "
              f"{label(c[12], 'l2_size'):>6s} {label(c[14], 'bp_type'):>12s}")

    # Also save as CSV
    import pandas as pd
    rows = []
    for i in range(len(X_opt)):
        row = {PARAM_NAMES[j]: int(X_opt[i, j]) for j in range(len(PARAM_NAMES))}
        row["pred_ipc"] = round(-F[i, 0], 4)
        row["pred_power"] = round(F[i, 1], 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    out_path = f"{args.model_dir}/pareto_front.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
