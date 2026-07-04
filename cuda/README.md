/usr/local/cuda-12.2/bin/nvcc -fatbin -O3 \
  -gencode arch=compute_61,code=sm_61 \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_86,code=compute_86 \
  cuda/pom_mine.cu \
  -o cuda/pom_mine_legacy.fatbin

/usr/local/cuda-13/bin/nvcc -fatbin -O3 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_100,code=sm_100 \
  -gencode arch=compute_120,code=sm_120 \
  -gencode arch=compute_89,code=compute_89 \
  cuda/pom_mine.cu \
  -o cuda/pom_mine_nextgen.fatbin
