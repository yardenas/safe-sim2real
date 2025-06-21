## Using this repositoy on Euler (ETH Zürich)
1. Use the provided `setup.bash` script via `source setup.bash`
2. Install environment `poetry install`


## Installing Madrona-MJX on Euler
1. Make sure that the steps above have been completed on the current terminal.
2. git clone https://github.com/shacklettbp/madrona_mjx.git
3. cd madrona_mjx
4. git submodule update --init --recursive
5. mkdir build
6. Load cmake `module load cmake 3.27.7`
7. Compile on a compute node ` srun --ntasks=1 --cpus-per-task=20 --gpus=rtx_4090:1 --time=0:30:00 --mem-per-cpu=10240 make -j`
8. Test `srun --ntasks=1 --cpus-per-task=20 --gpus=rtx_4090:1 --time=0:30:00 --mem-per-cpu=10240 python train_brax.py +experiment=cartpole_swingup_simple environment.task_params.vision=true`