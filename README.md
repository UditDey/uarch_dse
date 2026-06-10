# Automated CPU Microarchitecture Exploration using Surrogate Models

This repo contains an experimental [`gem5`](https://github.com/gem5/gem5) based microarchitecture Design Space Exploration pipeline

> [!WARNING]
> Work in progress

## Building
First download all the gem5 dependencies and set-up the Python `venv`
```bash
# gem5 build dependencies
sudo apt install -y \
    build-essential git python3 python3-dev python3-pip python3-venv \
    scons libprotobuf-dev protobuf-compiler libgoogle-perftools-dev \
    libboost-all-dev m4 zlib1g-dev libhdf5-dev pkg-config

# RISC-V cross-compiler (for building sqlite benchmark program)
sudo apt install -y gcc-riscv64-linux-gnu g++-riscv64-linux-gnu

# Python venv
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
Then build `gem5` and the `sqlite` benchmark program:
```bash
# Warning: will take a lot of time
make build_gem5

# Will be quicker
make build_sqlite
```

## Running the pipeline
TODO: Explain

Run these inside the `venv`
```bash
# Collect gem5 simulation run data
python dataset_collect.py --n-samples 300 --parallel 11 --gem5-bin third_party/gem5/build/RISCV/gem5.opt --gem5-config gem5_config.py --binary sqlite_bench --binary-args "80"

# Train surrogate XGBoost models
python train_surrogate.py dse_data.csv --model-dir models/

# Sketch Pareto Frontier from High IPC/High Power to Low IPC/Low Power
python dse_optimize.py --model-dir models/ --n-gen 300
```
