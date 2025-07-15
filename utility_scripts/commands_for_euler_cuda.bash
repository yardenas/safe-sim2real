module load cuda/12.4.1
module load cudnn/9.2.0.82-12
export PATH=/cluster/software/stacks/2024-06/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.2.0/cuda-12.4.1-k2chuuwwopbgkvfhnywfzo7hkemjnqjt/bin:$PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/cluster/software/stacks/2024-06/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.2.0/cuda-12.4.1-k2chuuwwopbgkvfhnywfzo7hkemjnqjt
