#!/bin/bash
# bsub -Is -W 0:30 -nnodes 1 -P MED106 $SHELL
# module load ibm-wml-ce/1.7.0-2
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 1 0 3 exec >run0.log 2>&1 &
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 1 0 summit-test exec >inter_jsrun.log 2>&1 &
# ----------------------------------------------
# This script executes training of kf.keras NN.
# ----------------------------------------------

# dump_prefix="/gpfs/alpine/scratch/apartin/med106"
# dump_prefix="/gpfs/alpine/med106/scratch/apartin"

device=$1
id=$2
global_sufx=$3

# dump_prefix="/gpfs/alpine/med106/scratch/$USER"
# gout="$dump_prefix"

gout="/gpfs/alpine/med106/scratch/$USER/$global_sufx"
mkdir -p $gout

export CUDA_VISIBLE_DEVICES=$device

DATAPATH="data/data.parquet"

echo "Using cuda device $device"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Run id: $id"
echo "Global output: $gout"

# python src/main.py --ii $id --ep $epoch --gout $gout > "$dump_prefix"/run"$id".log 2>&1
python src/main.py -dp $DATAPATH --gout $gout --ii $id > "$gout"/run"$id".log 2>&1

