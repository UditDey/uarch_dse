#!/usr/bin/env python3
"""
dse_collect.py — Collect gem5 simulation data across the design space
using Latin Hypercube Sampling for efficient coverage.

Usage:
    python3 dse_collect.py --n-samples 50 --gem5-bin third_party/gem5/build/RISCV/gem5.opt \
        --gem5-config dse_run.py --binary sqlite_bench --binary-args "50" \
        --output results/dse_data.csv

    # Parallel (4 concurrent gem5 runs):
    python3 dse_collect.py --n-samples 50 --parallel 4 ...
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import re
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# ── Design space definition ──────────────────────────────────────────
# Each parameter: (name, gem5_flag, [possible_values])
# LHS will sample uniformly across these discrete options.

DESIGN_SPACE = {
    "fetch_width":    ("--fetch-width",    [2, 4, 8]),
    "decode_width":   ("--decode-width",   [2, 4, 8]),
    "issue_width":    ("--issue-width",    [2, 4, 8]),
    "commit_width":   ("--commit-width",   [2, 4, 8]),
    "dispatch_width": ("--dispatch-width", [2, 4, 8]),
    "rob_entries":    ("--rob-entries",     [32, 64, 128, 192, 256]),
    "lq_entries":     ("--lq-entries",      [16, 32, 64]),
    "sq_entries":     ("--sq-entries",      [16, 32, 64]),
    "l1i_size":       ("--l1i-size",        ["16kB", "32kB", "64kB"]),
    "l1d_size":       ("--l1d-size",        ["16kB", "32kB", "64kB"]),
    "l1i_assoc":      ("--l1i-assoc",       [2, 4]),
    "l1d_assoc":      ("--l1d-assoc",       [2, 4]),
    "l2_size":        ("--l2-size",         ["128kB", "256kB", "512kB", "1MB"]),
    "l2_assoc":       ("--l2-assoc",        [4, 8, 16]),
    "bp_type":        ("--bp-type",         ["TournamentBP", "BiModeBP", "TAGE", "LocalBP"]),
}

# Constraints: widths should be consistent (decode >= issue, etc.)
# We enforce fetch >= decode >= issue after sampling.
WIDTH_PARAMS = ["fetch_width", "decode_width", "dispatch_width", "issue_width", "commit_width"]


def latin_hypercube_discrete(params, n_samples, rng):
    """
    Latin Hypercube Sampling for discrete parameter spaces.

    For each parameter with k levels, divides [0, k) into n_samples strata,
    assigns one sample per stratum, then maps to discrete values.
    """
    n_params = len(params)
    param_names = list(params.keys())
    param_values = [params[p][1] for p in param_names]  # list of value lists

    samples = np.zeros((n_samples, n_params), dtype=int)

    for j, values in enumerate(param_values):
        k = len(values)
        # Create permutation of strata indices
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            # Map stratum to a value index
            samples[i, j] = perm[i] % k

    # Convert indices to actual values
    configs = []
    for i in range(n_samples):
        config = {}
        for j, name in enumerate(param_names):
            config[name] = param_values[j][samples[i, j]]
        configs.append(config)

    return configs


def enforce_constraints(config):
    """
    Enforce microarchitectural constraints:
    - Pipeline widths: fetch >= decode >= dispatch >= issue, commit >= issue
    - LQ/SQ should be reasonable relative to ROB
    """
    widths = sorted([config[w] for w in WIDTH_PARAMS], reverse=True)
    config["fetch_width"] = widths[0]
    config["decode_width"] = widths[1]
    config["dispatch_width"] = widths[2]
    config["issue_width"] = widths[3]
    config["commit_width"] = widths[4]

    # commit width should be at least issue width
    config["commit_width"] = max(config["commit_width"], config["issue_width"])

    return config


def parse_stats(stats_path):
    """Parse gem5 stats.txt into dict."""
    stats = {}
    with open(stats_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("---") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    stats[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return stats


def S(stats, *names, default=0):
    for n in names:
        if n in stats:
            return int(stats[n])
    return int(default)


def compute_energy_proxy(stats, config, clock_mhz=2000):
    """
    Estimate power (in Watts) from gem5 stats + config.

    Energy model based on Horowitz ISSCC 2014 ("Computing's Energy Problem"),
    measured at 45nm 0.9V. Activity coefficients in picojoules (pJ).
    Power conversion: 1 pJ / 1 ps = 1 W.

    ┌─────────────────────────────────────────────────────────────────────┐
    │ MEASURED by Horowitz (45nm 0.9V):                                  │
    │   Int 32-bit add:           0.1 pJ                                 │
    │   Int 32-bit multiply:      3.1 pJ                                 │
    │   FP 32-bit add:            0.9 pJ                                 │
    │   FP 32-bit multiply/MAC:   4.6 pJ                                 │
    │   Register file (1kB, 64b): 6.0 pJ per access                     │
    │   8kB SRAM cache:          10.0 pJ per access                      │
    │   32kB SRAM cache:         20.0 pJ per access                      │
    │   1MB SRAM cache:         100.0 pJ per access                      │
    │   DRAM (64-bit):         1300.0 pJ per access                      │
    │                                                                     │
    │ ESTIMATED from SRAM scaling (energy ~ sqrt(size)):                  │
    │   ROB (~1.5kB):              4.0 pJ per read/write                 │
    │   Rename/RAT (~256B):        2.0 pJ per lookup                     │
    │   Issue queue (~512B):       2.0 pJ per wakeup/select              │
    │   Branch predictor (~1kB):   3.0 pJ per lookup                     │
    │   LSQ (~512B):               2.0 pJ per access                     │
    │                                                                     │
    │ STRUCTURAL (leakage + clock, per cycle):                           │
    │   Base core leakage:         2.0 pJ/cycle                          │
    │   Per-width overhead:        issue_width^2 * 0.5 pJ/cycle          │
    │     (bypass network, wakeup-select logic scale quadratically)       │
    │   Cache leakage:             proportional to size, ~0.001 pJ/B/cyc │
    │   ROB leakage:               ~0.01 pJ/entry/cycle                  │
    └─────────────────────────────────────────────────────────────────────┘

    Cache access energy interpolation (Horowitz anchors: 8kB=10pJ, 1MB=100pJ):
        E(size_bytes) = 10.0 * sqrt(size_bytes / 8192)
    """

    import math

    # Cycle time in picoseconds: 2GHz (2000 MHz) → 500 ps
    cycle_time_ps = 1e6 / clock_mhz

    # ── Helper: cache access energy from size ────────────────────────
    # Horowitz anchors: 8kB → 10 pJ, 1MB → 100 pJ
    # Fits E = 10 * sqrt(size / 8192), verified: sqrt(1048576/8192) ≈ 11.3 → ~113 pJ ≈ 100 pJ
    def cache_energy_per_access(size_bytes):
        return 10.0 * math.sqrt(size_bytes / 8192)

    # ── Parse cache sizes from config (string like "32kB" or int) ────
    def parse_size(val):
        if isinstance(val, (int, float)):
            return int(val)
        val = str(val)
        if "MB" in val or "MiB" in val:
            return int(''.join(c for c in val if c.isdigit())) * 1024 * 1024
        if "kB" in val or "KiB" in val:
            return int(''.join(c for c in val if c.isdigit())) * 1024
        return int(val)

    l1i_bytes = parse_size(config.get("l1i_size", "32kB"))
    l1d_bytes = parse_size(config.get("l1d_size", "64kB"))
    l2_bytes  = parse_size(config.get("l2_size", "256kB"))
    issue_w   = int(config.get("issue_width", 4))
    rob_sz    = int(config.get("rob_entries", 192))
    lq_sz     = int(config.get("lq_entries", 32))
    sq_sz     = int(config.get("sq_entries", 32))

    # ── Per-access energy for this config's caches (pJ) ──────────────
    E_L1I = cache_energy_per_access(l1i_bytes)   # e.g., 32kB → ~20 pJ
    E_L1D = cache_energy_per_access(l1d_bytes)   # e.g., 64kB → ~28 pJ
    E_L2  = cache_energy_per_access(l2_bytes)     # e.g., 256kB → ~56 pJ

    # ── Extract stats from gem5 ──────────────────────────────────────
    cycles    = S(stats, "system.cpu.numCycles")
    committed = S(stats, "system.cpu.commitStats0.numInsts",
                         "system.cpu.thread_0.numInsts")
    if committed == 0:
        return None, None, None

    # Instruction type breakdown
    def itype(t):
        return S(stats, f"system.cpu.commitStats0.committedInstType::{t}", default=0)

    alu_ops  = itype("IntAlu")
    mul_ops  = itype("IntMult")
    div_ops  = itype("IntDiv")
    fp_add   = itype("FloatAdd") + itype("FloatCmp") + itype("FloatCvt") + itype("FloatMisc")
    fp_mul   = itype("FloatMult") + itype("FloatMultAcc") + itype("FloatDiv") + itype("FloatSqrt")
    loads    = itype("MemRead") + itype("FloatMemRead")
    stores   = itype("MemWrite") + itype("FloatMemWrite")

    # If no type breakdown available, attribute all to ALU
    if alu_ops == 0 and mul_ops == 0 and fp_add == 0:
        alu_ops = committed

    # Pipeline structure accesses
    rob_acc    = S(stats, "system.cpu.rob.reads") + S(stats, "system.cpu.rob.writes")
    rename_acc = S(stats, "system.cpu.rename.lookups")
    issued     = S(stats, "system.cpu.instsIssued")
    bp_lookups = S(stats, "system.cpu.bac.branches")

    # Cache accesses
    ic_hit  = S(stats, "system.cpu.icache.demandHits::total")
    ic_miss = S(stats, "system.cpu.icache.demandMisses::total")
    dc_hit  = S(stats, "system.cpu.dcache.demandHits::total")
    dc_miss = S(stats, "system.cpu.dcache.demandMisses::total")
    l2_hit  = S(stats, "system.l2cache.demandHits::total")
    l2_miss = S(stats, "system.l2cache.demandMisses::total")

    # DRAM accesses
    mem_acc = (S(stats, "system.mem_ctrl.readReqs", "system.mem_ctrl.dram.readBursts", default=0) +
               S(stats, "system.mem_ctrl.writeReqs", "system.mem_ctrl.dram.writeBursts", default=0))

    # ── DYNAMIC ENERGY (per-access costs × access counts) ───────────

    # Execution units (Horowitz 45nm)
    E_exec = (
          0.1 * alu_ops          # Int ALU add: 0.1 pJ
        + 3.1 * mul_ops          # Int multiply: 3.1 pJ
        + 3.1 * div_ops          # Int divide: ~3.1 pJ (est. similar to multiply)
        + 0.9 * fp_add           # FP add/compare/convert: 0.9 pJ
        + 4.6 * fp_mul           # FP multiply/MAC/div/sqrt: 4.6 pJ
    )

    # Register file (Horowitz: 6 pJ per 64-bit access for ~1kB RF)
    # Approximate: 2 source reads + 1 dest write per instruction
    E_regfile = 6.0 * (committed * 3)

    # Pipeline bookkeeping (estimated from small SRAM sizes)
    E_pipeline = (
          4.0 * rob_acc           # ROB: ~1.5kB SRAM → ~4 pJ/access
        + 2.0 * rename_acc        # Rename/RAT: ~256B → ~2 pJ/lookup
        + 2.0 * issued            # Issue queue wakeup/select: ~512B → ~2 pJ
        + 3.0 * bp_lookups        # Branch predictor tables: ~1kB → ~3 pJ
    )

    # L1 caches (cost per access depends on configured size)
    E_l1 = (
          E_L1I * (ic_hit + ic_miss)   # every L1I access pays the per-access cost
        + E_L1D * (dc_hit + dc_miss)   # every L1D access pays the per-access cost
    )

    # L2 cache (accessed on every L1 miss)
    E_l2 = E_L2 * (l2_hit + l2_miss)

    # DRAM (Horowitz: 1300 pJ per 64-bit access)
    E_dram = 1300.0 * mem_acc

    # ── STRUCTURAL ENERGY (per-cycle costs × cycle count) ────────────
    # These represent leakage and clock distribution — energy spent
    # just by having the hardware, regardless of utilization.

    E_structural = (
        # Base core: clock tree, control logic, always-on circuits
        # ~2 pJ/cycle baseline for a minimal core
          2.0

        # Issue width overhead: bypass network and wakeup-select logic
        # scale quadratically with width (more ports, more wires)
        + 0.5 * (issue_w ** 2)

        # ROB leakage: proportional to number of entries
        + 0.01 * rob_sz

        # LSQ leakage
        + 0.01 * (lq_sz + sq_sz)

        # Cache leakage: proportional to total SRAM bytes
        # ~0.001 pJ per byte per cycle at 45nm
        + 0.001 * (l1i_bytes + l1d_bytes + l2_bytes)
    ) * cycles

    # ── TOTAL ────────────────────────────────────────────────────────
    energy_pJ = E_exec + E_regfile + E_pipeline + E_l1 + E_l2 + E_dram + E_structural

    ipc = committed / cycles if cycles > 0 else 0
    epi = energy_pJ / committed if committed > 0 else 0

    # Power in Watts: 1 pJ / 1 ps = 1 W
    # gem5 tick = 1 ps, so clock period in ps = config clock value
    power_W = energy_pJ / (cycles * cycle_time_ps) if cycles > 0 else 0

    return power_W, ipc, epi


def run_single(args_tuple):
    """Run a single gem5 simulation. Returns (config_dict, power_W, ipc, epi)."""
    idx, config, gem5_bin, gem5_config, binary, binary_args, results_dir = args_tuple

    outdir = os.path.join(results_dir, f"run_{idx:04d}")
    os.makedirs(outdir, exist_ok=True)

    # Build gem5 command
    cmd = [gem5_bin, f"--outdir={outdir}", gem5_config, "--binary", binary]
    if binary_args:
        cmd += ["--binary-args", binary_args]

    for param_name, (flag, _) in DESIGN_SPACE.items():
        cmd += [flag, str(config[param_name])]

    # Run gem5
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] run_{idx:04d}")
        return idx, config, None, None, None

    # Check for stats
    stats_path = os.path.join(outdir, "stats.txt")
    if not os.path.exists(stats_path) or os.path.getsize(stats_path) == 0:
        # Try dumping stats manually (some configs may not auto-dump)
        print(f"  [NO STATS] run_{idx:04d}")
        return idx, config, None, None, None

    stats = parse_stats(stats_path)
    power, ipc, epi = compute_energy_proxy(stats, config)

    if power is None:
        print(f"  [BAD STATS] run_{idx:04d}")
        return idx, config, None, None, None

    print(f"  run_{idx:04d}: IPC={ipc:.3f}  power={power:.3f}W  EPI={epi:.1f}  "
          f"w={config['issue_width']} ROB={config['rob_entries']} "
          f"L1D={config['l1d_size']} L2={config['l2_size']} BP={config['bp_type']}")

    return idx, config, power, ipc, epi


def main():
    ap = argparse.ArgumentParser(description="DSE data collection via LHS")
    ap.add_argument("--n-samples", type=int, default=50, help="Number of LHS samples")
    ap.add_argument("--gem5-bin", required=True, help="Path to gem5 binary")
    ap.add_argument("--gem5-config", required=True, help="Path to dse_run.py")
    ap.add_argument("--binary", required=True, help="Benchmark binary")
    ap.add_argument("--binary-args", default="", help="Benchmark arguments")
    ap.add_argument("--output", default="dse_data.csv", help="Output CSV path")
    ap.add_argument("--results-dir", default="results", help="Per-run output directory")
    ap.add_argument("--parallel", type=int, default=1, help="Number of parallel gem5 runs")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for LHS")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Generate LHS samples
    print(f"Generating {args.n_samples} LHS samples across {len(DESIGN_SPACE)} parameters...")
    configs = latin_hypercube_discrete(DESIGN_SPACE, args.n_samples, rng)

    # Enforce constraints
    configs = [enforce_constraints(c) for c in configs]

    # Deduplicate (constraints may collapse distinct samples)
    seen = set()
    unique_configs = []
    for c in configs:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)
    configs = unique_configs
    print(f"After dedup: {len(configs)} unique configs")

    os.makedirs(args.results_dir, exist_ok=True)

    # Build task list
    tasks = [
        (i, config, args.gem5_bin, args.gem5_config, args.binary,
         args.binary_args, args.results_dir)
        for i, config in enumerate(configs)
    ]

    # Run simulations
    print(f"\nRunning {len(tasks)} simulations (parallel={args.parallel})...\n")
    results = []

    if args.parallel <= 1:
        for task in tasks:
            results.append(run_single(task))
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(run_single, t): t for t in tasks}
            for future in as_completed(futures):
                results.append(future.result())

    # Sort by index
    results.sort(key=lambda x: x[0])

    # Write CSV
    fieldnames = list(DESIGN_SPACE.keys()) + ["power", "ipc", "epi"]
    successful = 0

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, config, power, ipc, epi in results:
            if power is not None:
                row = dict(config)
                row["power"] = round(power, 6)
                row["ipc"] = round(ipc, 6)
                row["epi"] = round(epi, 2)
                writer.writerow(row)
                successful += 1

    print(f"\nDone. {successful}/{len(tasks)} successful runs.")
    print(f"Dataset: {args.output}")
    print(f"Per-run stats: {args.results_dir}/run_XXXX/stats.txt")


if __name__ == "__main__":
    main()
