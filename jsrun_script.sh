#!/bin/bash
# bash run.sh 2 0 10
# bsub -Is -W 0:25 -nnodes 1 -P MED106 $SHELL
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_script.sh 0 0 5 exec >run0.log 2>&1
# module load ibm-wml-ce/1.7.0-2

# out_prefix="/gpfs/alpine/scratch/apartin/med106"
out_prefix="/gpfs/alpine/med106/scratch/$USER"
# out_prefix="/gpfs/alpine/med106/scratch/apartin"

device=$1
id=$2
epoch=$3
gout="$out_prefix"
echo "Using cuda device $device"
echo "Run id: $ii"
echo "Number of epochs: $epoch"
echo "Global output: $gout"

export CUDA_VISIBLE_DEVICES=$device

python main.py --ii $ii --ep $epoch --gout $gout exec > "$out_prefix"/run_"$id".log 2>&1
# python main.py --id $id --ep $epoch --gout $gout


