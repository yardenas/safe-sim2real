## Using this repositoy on Euler (ETH Zürich)
1. Use the provided `setup.bash` script via `source setup.bash`
2. Install environment `poetry install`


## Installing Madrona-MJX on Euler
1. Make sure that the steps above have been completed on the current terminal.
2. Go to the parent directory of the safe-sim2real repository: `cd ..`
3. `git clone https://github.com/shacklettbp/madrona_mjx.git && git checkout c34f3cf6d95148dba50ffeb981aea033b8a4d225`
4. `cd madrona_mjx`
5. `git submodule update --init --recursive`
6. Load cmake `module load cmake/3.27.7`
7. Point to cuda CMAKE witout loading cuda `export CMAKE_PREFIX_PATH="/cluster/software/stacks/2024-06/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.2.0/cuda-12.1.1-5znnrjb5x5xr26nojxp3yhh6v77il7ie/:${CMAKE_PREFIX_PATH}"`
8. Activate the poetry environment: `eval $(poetry env activate)`
9. `mkdir build && cd build && cmake -DLOAD_VULKAN=OFF .. && cd ..`
10. Compile on a compute node ` srun --ntasks=1 --cpus-per-task=20 --gpus=rtx_4090:1 --time=0:30:00 --mem-per-cpu=10240 make -j`
11. Change back to the root of the safe-sim2real repository: `cd .. && cd safe-sim2real`
12. `poetry install --with madrona-mjx`
13. Test `srun --ntasks=1 --cpus-per-task=20 --gpus=rtx_4090:1 --time=0:30:00 --mem-per-cpu=10240 python train_brax.py +experiment=cartpole_vision`