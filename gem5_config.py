"""
dse_run.py — Self-contained RISC-V O3CPU SE mode config for DSE.
Tested against gem5 v25.1.

    gem5.opt dse_run.py --binary ./benchmark
    gem5.opt dse_run.py --binary ./benchmark --rob-entries 128 --issue-width 4
"""

import argparse
import os
import m5
from m5.objects import *


# ── Inline cache classes (from learning_gem5 caches.py) ──────────────

class L1ICache(Cache):
    assoc = 2
    tag_latency = 2
    data_latency = 2
    response_latency = 2
    mshrs = 4
    tgts_per_mshr = 20

class L1DCache(Cache):
    assoc = 2
    tag_latency = 2
    data_latency = 2
    response_latency = 2
    mshrs = 4
    tgts_per_mshr = 20

class L2Cache(Cache):
    assoc = 8
    tag_latency = 20
    data_latency = 20
    response_latency = 20
    mshrs = 20
    tgts_per_mshr = 12


# ── Argument parsing ─────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="RISC-V O3CPU DSE config")

# Workload
parser.add_argument("--binary", required=True)
parser.add_argument("--binary-args", default="")

# Clock
parser.add_argument("--cpu-clock", default="2GHz")

# Pipeline widths
parser.add_argument("--fetch-width",    type=int, default=4)
parser.add_argument("--decode-width",   type=int, default=4)
parser.add_argument("--dispatch-width", type=int, default=4)
parser.add_argument("--issue-width",    type=int, default=4)
parser.add_argument("--commit-width",   type=int, default=4)

# OoO resources
parser.add_argument("--rob-entries",       type=int, default=192)
parser.add_argument("--lq-entries",        type=int, default=32)
parser.add_argument("--sq-entries",        type=int, default=32)
parser.add_argument("--num-phys-int-regs", type=int, default=256)
parser.add_argument("--num-phys-fp-regs",  type=int, default=256)

# Branch predictor
parser.add_argument("--bp-type", default="TournamentBP",
                    choices=["TournamentBP", "BiModeBP", "TAGE", "LocalBP"])

# Cache
parser.add_argument("--l1i-size",  default="16kB")
parser.add_argument("--l1d-size",  default="64kB")
parser.add_argument("--l1i-assoc", type=int, default=2)
parser.add_argument("--l1d-assoc", type=int, default=2)
parser.add_argument("--l2-size",   default="256kB")
parser.add_argument("--l2-assoc",  type=int, default=8)

args = parser.parse_args()


# ── System ───────────────────────────────────────────────────────────

system = System()
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = args.cpu_clock
system.clk_domain.voltage_domain = VoltageDomain()
system.mem_mode = "timing"
system.mem_ranges = [AddrRange("512MiB")]


# ── CPU ──────────────────────────────────────────────────────────────

system.cpu = RiscvO3CPU()

system.cpu.fetchWidth    = args.fetch_width
system.cpu.decodeWidth   = args.decode_width
system.cpu.dispatchWidth = args.dispatch_width
system.cpu.issueWidth    = args.issue_width
system.cpu.commitWidth   = args.commit_width

system.cpu.numROBEntries    = args.rob_entries
system.cpu.LQEntries        = args.lq_entries
system.cpu.SQEntries        = args.sq_entries
system.cpu.numPhysIntRegs   = args.num_phys_int_regs
system.cpu.numPhysFloatRegs = args.num_phys_fp_regs


# ── Branch predictor (v25.1 modular structure) ───────────────────────

bp_map = {
    "TournamentBP": TournamentBP,
    "BiModeBP":     BiModeBP,
    "TAGE":         TAGE,
    "LocalBP":      LocalBP,
}
system.cpu.branchPred = BranchPredictor(
    conditionalBranchPred=bp_map[args.bp_type]()
)


# ── L1 caches ────────────────────────────────────────────────────────

system.cpu.icache = L1ICache(size=args.l1i_size, assoc=args.l1i_assoc)
system.cpu.dcache = L1DCache(size=args.l1d_size, assoc=args.l1d_assoc)

system.cpu.icache.cpu_side = system.cpu.icache_port
system.cpu.dcache.cpu_side = system.cpu.dcache_port


# ── L2 bus + L2 cache ────────────────────────────────────────────────

system.l2bus = L2XBar()

system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports

system.l2cache = L2Cache(size=args.l2_size, assoc=args.l2_assoc)
system.l2cache.cpu_side = system.l2bus.mem_side_ports


# ── Memory bus + controller ──────────────────────────────────────────

system.membus = SystemXBar()

system.l2cache.mem_side = system.membus.cpu_side_ports

system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

system.system_port = system.membus.cpu_side_ports


# ── Interrupt controller ─────────────────────────────────────────────

system.cpu.createInterruptController()


# ── Workload ─────────────────────────────────────────────────────────

binary = os.path.abspath(args.binary)
system.workload = SEWorkload.init_compatible(binary)

process = Process()
process.cmd = [binary] + (args.binary_args.split() if args.binary_args else [])
system.cpu.workload = process
system.cpu.createThreads()


# ── Run ──────────────────────────────────────────────────────────────

root = Root(full_system=False, system=system)
m5.instantiate()

print(f"Running: {' '.join(process.cmd)}")
print(f"Config: width={args.issue_width} ROB={args.rob_entries} "
      f"L1D={args.l1d_size} L2={args.l2_size} BP={args.bp_type}")

exit_event = m5.simulate()
m5.stats.dump()
print(f"Exiting @ tick {m5.curTick()} because {exit_event.getCause()}")
