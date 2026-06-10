#!/usr/bin/env python3
"""
dse_optimize.py — Find Pareto-optimal configs using saved XGBoost surrogates.

Uses pymoo's mixed-variable support so every parameter is a proper
discrete Choice — no continuous relaxation, no snap-to-grid bias.

Width constraints (fetch >= decode >= dispatch >= issue, commit >= issue)
are enforced inside the evaluator so the surrogate always sees valid
configs, and again when extracting the Pareto front for output.

Usage:
    python3 dse_optimize.py
    python3 dse_optimize.py --model-dir models/ --n-gen 300
"""

import argparse
import numpy as np
import pandas as pd
import xgboost as xgb

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Choice
from pymoo.core.mixed import MixedVariableGA
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.optimize import minimize
from pymoo.termination import get_termination


# ── Design space (matches dse_collect.py & train_surrogate.py) ───────
#
# Every parameter is a discrete Choice.  Sizes and bp_type are stored
# in the surrogate's encoded form (bytes / int label) so we can feed
# the model directly without a separate encode step.

SIZE_MAP = {16384: "16kB", 32768: "32kB", 65536: "64kB",
            131072: "128kB", 262144: "256kB", 524288: "512kB",
            1048576: "1MB"}
BP_MAP = {0: "LocalBP", 1: "BiModeBP", 2: "TournamentBP", 3: "TAGE"}

# Ordered list — the surrogate expects features in exactly this order.
PARAM_NAMES = [
    "fetch_width", "decode_width", "issue_width", "commit_width",
    "dispatch_width", "rob_entries", "lq_entries", "sq_entries",
    "l1i_size", "l1d_size", "l1i_assoc", "l1d_assoc",
    "l2_size", "l2_assoc", "bp_type",
]

PARAM_OPTIONS = {
    "fetch_width":    [2, 4, 8],
    "decode_width":   [2, 4, 8],
    "issue_width":    [2, 4, 8],
    "commit_width":   [2, 4, 8],
    "dispatch_width": [2, 4, 8],
    "rob_entries":    [32, 64, 128, 192, 256],
    "lq_entries":     [16, 32, 64],
    "sq_entries":     [16, 32, 64],
    "l1i_size":       [16384, 32768, 65536],
    "l1d_size":       [16384, 32768, 65536],
    "l1i_assoc":      [2, 4],
    "l1d_assoc":      [2, 4],
    "l2_size":        [131072, 262144, 524288, 1048576],
    "l2_assoc":       [4, 8, 16],
    "bp_type":        [0, 1, 2, 3],
}


# ── Width constraint helpers ─────────────────────────────────────────

def _snap(value, options):
    """Return the closest value in `options`."""
    arr = np.array(options)
    return options[int(np.argmin(np.abs(arr - value)))]


def fix_widths(xd):
    """
    Enforce fetch >= decode >= dispatch >= issue, commit >= issue.

    Sorts the five width values descending and snaps each back to
    a legal Choice value.  Operates on a dict in-place and returns it.
    """
    widths = sorted(
        [xd["fetch_width"], xd["decode_width"], xd["dispatch_width"],
         xd["issue_width"], xd["commit_width"]],
        reverse=True,
    )
    xd["fetch_width"]    = _snap(widths[0], PARAM_OPTIONS["fetch_width"])
    xd["decode_width"]   = _snap(widths[1], PARAM_OPTIONS["decode_width"])
    xd["dispatch_width"] = _snap(widths[2], PARAM_OPTIONS["dispatch_width"])
    xd["issue_width"]    = _snap(widths[3], PARAM_OPTIONS["issue_width"])
    xd["commit_width"]   = _snap(
        max(widths[4], xd["issue_width"]),
        PARAM_OPTIONS["commit_width"],
    )
    return xd


# ── Problem definition ───────────────────────────────────────────────

class DSEProblem(ElementwiseProblem):
    """
    Bi-objective: maximise IPC (minimise −IPC), minimise power.

    Each variable is a Choice over its legal discrete values.
    Width constraints are enforced before querying the surrogate
    so the model always sees physically valid configurations.
    """

    def __init__(self, ipc_model, power_model, **kwargs):
        variables = {
            name: Choice(options=opts)
            for name, opts in PARAM_OPTIONS.items()
        }
        super().__init__(vars=variables, n_obj=2, **kwargs)
        self.ipc_model = ipc_model
        self.power_model = power_model

    def _evaluate(self, X, out, *args, **kwargs):
        # Fix widths before querying surrogate.
        xd = fix_widths(dict(X))

        # Build feature vector in the order the surrogate expects.
        x = np.array([[xd[p] for p in PARAM_NAMES]], dtype=float)
        ipc = float(self.ipc_model.predict(x)[0])
        power = float(self.power_model.predict(x)[0])
        out["F"] = [-ipc, power]


# ── Pretty-printing helpers ──────────────────────────────────────────

def label(val, name):
    if name in ("l1i_size", "l1d_size", "l2_size"):
        return SIZE_MAP.get(int(val), str(int(val)))
    if name == "bp_type":
        return BP_MAP.get(int(val), str(int(val)))
    return str(int(val))


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--n-gen", type=int, default=200)
    ap.add_argument("--pop-size", type=int, default=100)
    args = ap.parse_args()

    # Load surrogate models
    ipc_model = xgb.XGBRegressor()
    ipc_model.load_model(f"{args.model_dir}/ipc.json")
    power_model = xgb.XGBRegressor()
    power_model.load_model(f"{args.model_dir}/power.json")
    print("Loaded surrogate models\n")

    problem = DSEProblem(ipc_model, power_model)

    # MixedVariableGA with NSGA-II survival for multi-objective.
    # Default MixedVariableDuplicateElimination handles dict-based
    # individuals correctly.
    algorithm = MixedVariableGA(
        pop_size=args.pop_size,
        survival=RankAndCrowdingSurvival(),
    )

    print(f"Running NSGA-II (mixed-variable): "
          f"pop={args.pop_size}  gen={args.n_gen}")
    result = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", args.n_gen),
        seed=42,
        verbose=False,
    )

    # ── Extract Pareto front ─────────────────────────────────────────
    F = result.F                  # (n_solutions, 2): [-ipc, power]
    X_raw = result.X              # list of dicts (mixed-variable output)

    # Apply width fix to output (evaluator fixed a copy; result.X is raw)
    X_dicts = [fix_widths(dict(xd)) for xd in X_raw]

    # Deduplicate (fixed widths may collapse distinct raw configs)
    seen = set()
    keep = []
    for i, xd in enumerate(X_dicts):
        key = tuple(xd[p] for p in PARAM_NAMES)
        if key not in seen:
            seen.add(key)
            keep.append(i)
    X_dicts = [X_dicts[i] for i in keep]
    F = F[keep]

    # Sort by IPC descending (F[:,0] is -IPC, so ascending sort)
    order = np.argsort(F[:, 0])
    X_dicts = [X_dicts[i] for i in order]
    F = F[order]

    # ── Print ────────────────────────────────────────────────────────
    print(f"\nPareto front: {len(X_dicts)} configs\n")
    header = (f"{'#':>3s} {'IPC':>7s} {'Power(W)':>10s}   "
              f"{'fw':>3s} {'dw':>3s} {'dpw':>3s} {'iw':>3s} {'cw':>3s} "
              f"{'ROB':>4s} {'LQ':>3s} {'SQ':>3s} "
              f"{'L1I':>5s} {'L1D':>5s} {'L1Ia':>4s} {'L1Da':>4s} "
              f"{'L2':>6s} {'L2a':>3s} {'BP':>12s}")
    print(header)
    print("-" * len(header))

    for i, xd in enumerate(X_dicts):
        ipc = -F[i, 0]
        power = F[i, 1]
        print(
            f"{i+1:3d} {ipc:7.4f} {power:10.4f}   "
            f"{int(xd['fetch_width']):3d} {int(xd['decode_width']):3d} "
            f"{int(xd['dispatch_width']):3d} {int(xd['issue_width']):3d} "
            f"{int(xd['commit_width']):3d} "
            f"{int(xd['rob_entries']):4d} "
            f"{int(xd['lq_entries']):3d} {int(xd['sq_entries']):3d} "
            f"{label(xd['l1i_size'], 'l1i_size'):>5s} "
            f"{label(xd['l1d_size'], 'l1d_size'):>5s} "
            f"{int(xd['l1i_assoc']):4d} {int(xd['l1d_assoc']):4d} "
            f"{label(xd['l2_size'], 'l2_size'):>6s} "
            f"{int(xd['l2_assoc']):3d} "
            f"{label(xd['bp_type'], 'bp_type'):>12s}"
        )

    # ── Save CSV ─────────────────────────────────────────────────────
    rows = []
    for i, xd in enumerate(X_dicts):
        row = {}
        for p in PARAM_NAMES:
            v = xd[p]
            # Decode to human-readable for the CSV
            if p in ("l1i_size", "l1d_size", "l2_size"):
                row[p] = SIZE_MAP.get(int(v), str(int(v)))
            elif p == "bp_type":
                row[p] = BP_MAP.get(int(v), str(int(v)))
            else:
                row[p] = int(v)
        row["pred_ipc"] = round(-F[i, 0], 4)
        row["pred_power"] = round(F[i, 1], 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = f"{args.model_dir}/pareto_front.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
