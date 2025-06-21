export WANDB_API_KEY='<your-key>'
export XLA_FLAGS=--xla_gpu_triton_gemm_any=true                                                     │B
export WANDB_CACHE_DIR=/cluster/scratch/yardas/wandb                                                │I
module load stack/2024-06                                                                           │D
module load gcc/12.2.0                                                                              │
module load eth_proxy                                                                               │P

# Madrona dependencies                                                                              │R
module load libx11/1.8.4-ns5x2da                                                                    │T
module load libxrandr/1.5.3-acspwjp                                                                 │I
module load libxinerama/1.1.3                                                                       │T
module load libxcursor/1.2.1                                                                        │I
module load libxi/1.7.6-qeazdpn                                                                     │O
module load mesa/23.0.3                                                                             │N

# Hidden Madrona dependencies :^)                                                                   │
module load libxrender/0.9.10-kss2t7k                                                               │
module load libxext/1.3.3-e74gj2z                                                                   │
module load libxfixes/5.0.2-5fbeidb                                                                 │

module load python/3.11.6 