#!/usr/bin/env bash
# Shippable `libkeryx-llama.so` — the miner's in-process llama.cpp engine (Phase 2
# in-process engine: hosts the model for the PoM walk zero-dup + serves OPoI inference).
# Built inside the glibc-2.31 container; links the SAME CUDA line as the package it ships in
# (cublas/cudart resolve from the package's lib/ at runtime).
#
# Usage: build-keryx-llama.sh <modern|legacy|pascal|no-avx|no-avx-no-flash> [JOBS]
#   modern: container CUDA 12.9, archs 75;80;86;89;90;120
#   legacy: /tmp/cuda124 (12.4), archs 70;75;80;86;89;90
#   pascal: /tmp/cuda124 (12.4), archs 60;61
#   no-avx: compatibility build with AVX/FMA/CET disabled; flash-attention remains enabled by default
#   no-avx-no-flash: compatibility build with AVX/FMA/CET disabled and CUDA flash-attention explicitly disabled
# Output: hiveos/dist-<line>/libkeryx-llama.so  (package-line.sh bundles it when present)
# Optional toggle: set KERYX_FLASH_ATTN=0 to disable CUDA flash-attention kernels for any build; default is enabled.
# NOTE: llama.cpp b10015 uses GGML_CUDA_FA / GGML_CUDA_FA_ALL_QUANTS CMake options.
#
# llama.cpp PINNED to b10015 — the SAME pin as build-llama-server.sh and the byte-identity
# proof in tools/llama_zerodup_spike. Bump all together, then re-verify the spike.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LINE="$1"
JOBS="${2:-16}"
TAG=b10015

case "$LINE" in
modern)
	ARCHS="75;80;86;89;90;120"
	CUDAMOUNT=()
	KCUDA=/usr/local/cuda
	;;
# Broad compatibility set capped at sm86 to avoid compute_89 host-stub invalid-opcode traps
legacy)
	ARCHS="61-real;70-real;75-real;80-real;86-real;86-virtual"
	CUDAMOUNT=(-v /tmp/cuda124:/opt/cuda:ro)
	KCUDA=/opt/cuda
	;;
pascal)
	ARCHS="60;61"
	CUDAMOUNT=(-v /tmp/cuda124:/opt/cuda:ro)
	KCUDA=/opt/cuda
	;;
no-avx)
	ARCHS="61-real;70-real;75-real;80-real;86-real;86-virtual"
	CUDAMOUNT=(-v /tmp/cuda124:/opt/cuda:ro)
	KCUDA=/opt/cuda
	CPUFLAGS="-DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_AVX_VNNI=OFF -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF"
	CMAKE_FLAGS="-march=x86-64 -mtune=generic -mno-avx -mno-avx2 -mno-fma -mno-f16c -mno-avx512f -mno-avx512vl -mno-avx512bw -mno-avx512dq -mno-avx512vnni -mno-avx512bf16 -fcf-protection=none -falign-functions=1 -falign-jumps=1 -falign-labels=1 -falign-loops=1"
	WRAPFLAGS="-march=nehalem"
	;;
no-avx-no-flash)
	ARCHS="61-real;70-real;75-real;80-real;86-real;86-virtual"
	CUDAMOUNT=(-v /tmp/cuda124:/opt/cuda:ro)
	KCUDA=/opt/cuda
	CPUFLAGS="-DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_AVX_VNNI=OFF -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF -DGGML_CUDA_FA=OFF -DGGML_CUDA_FA_ALL_QUANTS=OFF"
	CMAKE_FLAGS="-march=x86-64 -mtune=generic -mno-avx -mno-avx2 -mno-fma -mno-f16c -mno-avx512f -mno-avx512vl -mno-avx512bw -mno-avx512dq -mno-avx512vnni -mno-avx512bf16 -fcf-protection=none -falign-functions=1 -falign-jumps=1 -falign-labels=1 -falign-loops=1"
	WRAPFLAGS="-march=nehalem"
	;;
*)
	echo "usage: $0 <modern|legacy|pascal|no-avx|no-avx-no-flash> [JOBS]"
	exit 1
	;;
esac

: "${CPUFLAGS:=}"
: "${CMAKE_FLAGS:=}"
: "${WRAPFLAGS:=}"
# NVCC-generated host stubs (from .cu files) need explicit host-compiler flags too;
# otherwise some builds can still emit unsupported CPU opcodes on old x86 hosts.
: "${CMAKE_CUDA_FLAGS:=}"
if [ -z "${CMAKE_CUDA_FLAGS}" ] && [ -n "${CMAKE_FLAGS}" ]; then
	CUDA_HOST_FLAGS="${CMAKE_FLAGS// /,}"
	CMAKE_CUDA_FLAGS="--compiler-options=${CUDA_HOST_FLAGS}"
fi
if [ "${KERYX_FLASH_ATTN:-1}" != "1" ]; then
	CPUFLAGS="$CPUFLAGS -DGGML_CUDA_FA=OFF -DGGML_CUDA_FA_ALL_QUANTS=OFF"
fi

OUT="$REPO/hiveos/dist-$LINE"
mkdir -p "$OUT"
SRC=/tmp/llama-src-$TAG
if [ ! -d "$SRC" ]; then
	git clone --quiet --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$SRC"
fi

docker run --rm --network host \
	-v "$SRC":/llama -v "$REPO":/repo:ro -v "$OUT":/out "${CUDAMOUNT[@]}" \
	-e KCUDA="$KCUDA" -e ARCHS="$ARCHS" -e JOBS="$JOBS" -e CPUFLAGS="$CPUFLAGS" -e CMAKE_FLAGS="$CMAKE_FLAGS" -e CMAKE_CUDA_FLAGS="$CMAKE_CUDA_FLAGS" -e WRAPFLAGS="$WRAPFLAGS" -e KERYX_SPIKE="${KERYX_SPIKE:-0}" \
	keryx-build:offline bash -euo pipefail -c '
    if [ ! -x /tmp/cmk/bin/cmake ]; then
      curl -sL https://github.com/Kitware/CMake/releases/download/v3.28.6/cmake-3.28.6-linux-x86_64.tar.gz \
        | tar xz -C /tmp && mv /tmp/cmake-3.28.6-linux-x86_64 /tmp/cmk
    fi
    export PATH=/tmp/cmk/bin:$KCUDA/bin:$PATH CUDA_HOME=$KCUDA
    B=/tmp/llama-pic-build
    /tmp/cmk/bin/cmake -S /llama -B $B -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF $CPUFLAGS -DGGML_CUDA_NCCL=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=$KCUDA/bin/nvcc \
      -DCMAKE_C_FLAGS="$CMAKE_FLAGS" -DCMAKE_CXX_FLAGS="$CMAKE_FLAGS" -DCMAKE_CUDA_FLAGS="$CMAKE_CUDA_FLAGS"
    /tmp/cmk/bin/cmake --build $B --target llama -j "$JOBS"
    g++ -O2 -std=c++17 -shared -fPIC -fopenmp -fcf-protection=none $WRAPFLAGS /repo/tools/keryx-llama/keryx_llama.cpp \
      -I /llama/include -I /llama/ggml/include -I /llama/src -I /llama/common \
      -I $KCUDA/include \
      -Wl,--start-group $B/src/libllama.a $B/ggml/src/ggml-cuda/libggml-cuda.a \
        $B/ggml/src/libggml-cpu.a $B/ggml/src/libggml.a $B/ggml/src/libggml-base.a \
      -Wl,--end-group \
      -L$KCUDA/lib64 -L$KCUDA/targets/x86_64-linux/lib -lcudart -lcublas -lcublasLt \
      -L$KCUDA/lib64/stubs -L$KCUDA/targets/x86_64-linux/lib/stubs -lcuda \
      -lpthread -ldl -o /out/libkeryx-llama.so
    chmod a+rx /out/libkeryx-llama.so

    # Byte-identity spike (KERYX_SPIKE=1): proves the llama-resident tensor bytes are byte-identical
    # to the on-disk GGUF (canonical name-sorted order — what the PoM walk gathers and R_T pins).
    # Reuses the static libs already built in $B. Run: ./spike <model.gguf> <gpu_ordinal>; exit 0 = OK.
    if [ "${KERYX_SPIKE:-0}" = "1" ]; then
      g++ -O2 -std=c++17 -fcf-protection=none $WRAPFLAGS /repo/tools/llama_zerodup_spike/spike.cpp \
        -I /llama/include -I /llama/ggml/include -I /llama/src \
        -I $KCUDA/include \
        -Wl,--start-group $B/src/libllama.a $B/ggml/src/ggml-cuda/libggml-cuda.a \
          $B/ggml/src/libggml-cpu.a $B/ggml/src/libggml.a $B/ggml/src/libggml-base.a \
        -Wl,--end-group \
        -L$KCUDA/lib64 -L$KCUDA/targets/x86_64-linux/lib -lcudart -lcublas -lcublasLt \
        -L$KCUDA/lib64/stubs -L$KCUDA/targets/x86_64-linux/lib/stubs -lcuda \
        -lpthread -ldl -o /out/spike
      chmod a+rx /out/spike
    fi
  '

if [ -f "$OUT/libkeryx-llama.so" ]; then
	size=$(stat -c %s "$OUT/libkeryx-llama.so")
	glibc=$(objdump -T "$OUT/libkeryx-llama.so" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -V | tail -1)
	syms=$(nm -D "$OUT/libkeryx-llama.so" | grep -c keryx_llama)
	echo ">> $LINE libkeryx-llama.so: $size bytes, glibc=$glibc, syms=$syms"
else
	echo ">> $LINE libkeryx-llama.so: missing"
fi
