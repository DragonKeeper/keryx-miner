# Cuda Support For Keryx-Miner

## Manual fatbin workflow (single final binary)

The CUDA worker now tries embedded fatbins first, per GPU, and falls back to
the existing PTX routing only if the fatbin cannot be loaded.


## Building

This version includes precompiled PTX and fatbins, which should work with most
modern GPUs. To compile the PTX and fatbins yourself, follow these commands:

```sh
git clone https://github.com/Keryx-Labs/keryx-miner.git

cd keryx-miner

# Run each command separately.

#PTX Building

#Compute version 6.1
/usr/local/cuda-12.2/bin/nvcc \
  plugins/cuda/kaspa-cuda-native/src/kaspa-cuda.cu \
  -std=c++11 -O3 --restrict --ptx \
  --gpu-architecture=compute_61 \
  --gpu-code=sm_61 \
  -o plugins/cuda/resources/keryx-cuda-sm61.ptx \
  -Xptxas -O3 -Xcompiler -O3

#Compute version 7.0
/usr/local/cuda-12.2/bin/nvcc \
  plugins/cuda/kaspa-cuda-native/src/kaspa-cuda.cu \
  -std=c++11 -O3 --restrict --ptx \
  --gpu-architecture=compute_70 \
  --gpu-code=sm_70 \
  -o plugins/cuda/resources/keryx-cuda-sm70.ptx \
  -Xptxas -O3 -Xcompiler -O3

#Compute version 7.5
/usr/local/cuda-12.2/bin/nvcc \
  plugins/cuda/kaspa-cuda-native/src/kaspa-cuda.cu \
  -std=c++11 -O3 --restrict --ptx \
  --gpu-architecture=compute_75 \
  --gpu-code=sm_75 \
  -o plugins/cuda/resources/keryx-cuda-sm75.ptx \
  -Xptxas -O3 -Xcompiler -O3

#Compute version 8.6
/usr/local/cuda-12.2/bin/nvcc \
  plugins/cuda/kaspa-cuda-native/src/kaspa-cuda.cu \
  -std=c++11 -O3 --restrict --ptx \
  --gpu-architecture=compute_86 \
  --gpu-code=sm_86 \
  -o plugins/cuda/resources/keryx-cuda-sm86.ptx \
  -Xptxas -O3 -Xcompiler -O3

#compute version 8.9
/usr/local/cuda-13/bin/nvcc \
  plugins/cuda/kaspa-cuda-native/src/kaspa-cuda.cu \
  -std=c++11 -O3 --restrict --ptx \
  --gpu-architecture=compute_89 \
  --gpu-code=sm_89 \
  -o plugins/cuda/resources/keryx-cuda-sm89.ptx \
  -Xptxas -O3 -Xcompiler -O3

#compute version 9.0
/usr/local/cuda-13/bin/nvcc \
  plugins/cuda/kaspa-cuda-native/src/kaspa-cuda.cu \
  -std=c++11 -O3 --restrict --ptx \
  --gpu-architecture=compute_89 \
  --gpu-code=sm_90 \
  -o plugins/cuda/resources/keryx-cuda-sm90.ptx \
  -Xptxas -O3 -Xcompiler -O3

  #compute version 10.0
/usr/local/cuda-13/bin/nvcc \
  plugins/cuda/kaspa-cuda-native/src/kaspa-cuda.cu \
  -std=c++11 -O3 --restrict --ptx \
  --gpu-architecture=compute_100 \
  --gpu-code=sm_89 \
  -o plugins/cuda/resources/keryx-cuda-sm100.ptx \
  -Xptxas -O3 -Xcompiler -O3

  #compute version 12.0
/usr/local/cuda-13/bin/nvcc \
  plugins/cuda/kaspa-cuda-native/src/kaspa-cuda.cu \
  -std=c++11 -O3 --restrict --ptx \
  --gpu-architecture=compute_120 \
  --gpu-code=sm_89 \
  -o plugins/cuda/resources/keryx-cuda-sm120.ptx \
  -Xptxas -O3 -Xcompiler -O3

# Fatbin building

#Legacy Fatbin
/usr/local/cuda-12.2/bin/nvcc -fatbin -O3 -std=c++11 --restrict \
  -gencode arch=compute_61,code=sm_61 \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_86,code=compute_86 \
  plugins/cuda/kaspa-cuda-native/src/kaspa-cuda.cu \
  -o plugins/cuda/resources/keryx-legacy.fatbin

#Nextgen Fatbin
/usr/local/cuda-13/bin/nvcc -fatbin -O3 -std=c++11 --restrict \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_100,code=sm_100 \
  -gencode arch=compute_120,code=sm_120 \
  -gencode arch=compute_89,code=compute_89 \
  plugins/cuda/kaspa-cuda-native/src/kaspa-cuda.cu \
  -o plugins/cuda/resources/keryx-nextgen.fatbin

```

The plugin is a shared library that lives alongside the miner binary.
You can build the library and binary by running:

```sh
cargo build --release -p keryx-miner -p keryxcuda
```
