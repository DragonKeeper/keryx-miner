# PoM Fatbin Build Instructions

This directory contains the CUDA source for PoM mining kernel builds.

The miner expects two prebuilt PoM fatbins:

- `cuda/pom_mine_legacy.fatbin`
- `cuda/pom_mine_nextgen.fatbin`

These are embedded at build time and loaded at runtime with a per-GPU fatbin-first ladder.

## Prerequisites

- CUDA 12.2 `nvcc` available (legacy fatbin build)
- CUDA 13.x `nvcc` available (nextgen fatbin build)
- Run commands from repository root

## Manual Build

### 1) Build legacy PoM fatbin (CUDA 12.2)

```sh
/usr/local/cuda-12.2/bin/nvcc -fatbin -O3 \
  -gencode arch=compute_61,code=sm_61 \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_86,code=compute_86 \
  cuda/pom_mine.cu \
  -o cuda/pom_mine_legacy.fatbin
```

### 2) Build nextgen PoM fatbin (CUDA 13.x)

```sh
/usr/local/cuda-13/bin/nvcc -fatbin -O3 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_100,code=sm_100 \
  -gencode arch=compute_120,code=sm_120 \
  -gencode arch=compute_89,code=compute_89 \
  cuda/pom_mine.cu \
  -o cuda/pom_mine_nextgen.fatbin
```

## Verify Outputs

```sh
ls -lh cuda/pom_mine_legacy.fatbin cuda/pom_mine_nextgen.fatbin
```

Both files must exist and be non-empty.

## Rebuild Miner

```sh
export PATH="/usr/local/cuda-12.2/bin:$PATH"
export CUDA_HOME="/usr/local/cuda-12.2"
export CUDA_ROOT="/usr/local/cuda-12.2"
export CUDA_PATH="/usr/local/cuda-12.2"

cargo build --release -p keryx-miner
```

## Runtime Check

At startup, look for PoM logs that show per-GPU selection, for example:

- `startup loaded nextgen fatbin`
- `startup loaded legacy fatbin`

If fatbins are not available/compatible on a card, runtime falls back to the PoM PTX ladder.
