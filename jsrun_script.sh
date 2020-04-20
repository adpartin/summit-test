#!/bin/bash
# bsub -Is -W 0:25 -nnodes 1 -P MED106 $SHELL
# module load ibm-wml-ce/1.7.0-2
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_script.sh 0 0 3 exec >run0.log 2>&1

# dump_prefix="/gpfs/alpine/scratch/apartin/med106"
# dump_prefix="/gpfs/alpine/med106/scratch/apartin"
dump_prefix="/gpfs/alpine/med106/scratch/$USER"
# move_prefix=""
# mkdir -p "$move_prefix"/save

device=$1
ii=$2
epoch=$3
gout="$dump_prefix"
echo "Using cuda device $device"
echo "Run id: $ii"
echo "Number of epochs: $epoch"
echo "Global output: $gout"

export CUDA_VISIBLE_DEVICES=$device

# Works
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_script.sh 0 0 3
# python main.py --ii $ii --ep $epoch --gout $gout # exec > "$dump_prefix"/run_"$ii".log 2>&1

# Works
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_script.sh 0 1 3
python main.py --ii $ii --ep $epoch --gout $gout > "$dump_prefix"/run"$ii".log 2>&1

# Doesn't work TODO 
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_script.sh 0 2 3
# python main.py --ii $ii --ep $epoch --gout $gout exec > "$dump_prefix"/run"$ii".log 2>&1




