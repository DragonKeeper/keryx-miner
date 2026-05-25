# Keryx Miner

A high-performance miner for **Keryx**, combining GPU PoW (kHeavyHash) with on-chain AI inference (OPoI — Optimistic Proof of Inference).

---

## Precompiled Binaries

Download the latest release from the [Releases page](https://github.com/Keryx-Labs/keryx-miner/releases).

---

## Build from Source

### Standard build (PoW only, no inference)

Requires: Rust + Cargo ([rustup.rs](https://rustup.rs/)), `protoc` (`protobuf-compiler`)

```bash
git clone https://github.com/Keryx-Labs/keryx-miner.git
cd keryx-miner
cargo build --release --bin keryx-miner
```

Binary: `target/release/keryx-miner`

---

### CUDA build (PoW + GPU inference)

The inference engine (candle) requires **CUDA ≤ 12.6**.

#### Option A — CUDA 12.6 installed on host

```bash
cd keryx-miner
CUDA_COMPUTE_CAP=86 CUDA_PATH=/usr/local/cuda cargo build --release --bin keryx-miner
```

Binary: `target/release/keryx-miner`

#### Option B — CUDA 13.x or incompatible gcc on host (build via container)

If your system has CUDA 13.x or gcc 15+ (e.g. Fedora 40+, Ubuntu 25+), build inside a CUDA 12.6 container. The binary runs on the host via driver forward-compatibility.

Requires: [Podman](https://podman.io/) (rootless) or Docker, NVIDIA driver ≥ 530.

```bash
cd keryx-miner
podman run --rm --security-opt label=disable \
  -v "$PWD":/src -w /src \
  -e CUDA_COMPUTE_CAP=86 \
  -e CARGO_TARGET_DIR=/src/target-cuda \
  docker.io/nvidia/cuda:12.6.3-devel-ubuntu24.04 \
  bash -c '
    apt-get update -qq && apt-get install -y -qq \
      curl build-essential pkg-config libssl-dev ca-certificates protobuf-compiler >/dev/null 2>&1
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
    . "$HOME/.cargo/env"
    export CUDA_PATH=/usr/local/cuda PROTOC=/usr/bin/protoc
    cargo build --release --bin keryx-miner'
```

Binary: `target-cuda/release/keryx-miner`

> `libcuda.so.1` (driver) is the only runtime dependency — no need to ship libcudart or libcublas.

**CUDA_COMPUTE_CAP by GPU generation:**

| GPU generation | Compute cap |
|----------------|-------------|
| RTX 30xx (Ampere) | `86` |
| RTX 40xx (Ada Lovelace) | `89` |
| RTX 50xx (Blackwell) | `100` |

---

## Usage

```bash
./keryx-miner --mining-address keryx:YOUR_ADDRESS
```

### Inference tiers (OPoI)

| Flag | Models supported | Min VRAM |
|------|-----------------|----------|
| *(none)* | TinyLlama 1.1B + DeepSeek-R1-8B | 8 GB |
| `--light` | TinyLlama 1.1B only | 4 GB |
| `--high` | TinyLlama 1.1B + DeepSeek-R1-8B + DeepSeek-R1-32B | 24 GB |
| `--very-high` | All 4 models (+ LLaMA-3.3-70B) | 32 GB |

Models are loaded **on demand** when a request arrives and cached between requests. Mining pauses during inference, then resumes automatically.

To run without inference (PoW only):

```bash
./keryx-miner --mining-address keryx:YOUR_ADDRESS --no-opoi
```

### All options

```bash
./keryx-miner --help
```

---

## Connect

* **Website:** [keryx-labs.com](https://keryx-labs.com)
* **X (Twitter):** [@Keryx_Labs](https://x.com/Keryx_Labs)
* **Discord:** [Join the Community](https://discord.gg/U9eDmBUKTF)

---

> "Intelligence is the message. Keryx is the messenger."

---

## Dev Fund

2% of mining rewards support development by default.

```bash
--devfund-percent XX.YY
```
