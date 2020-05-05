#!/bin/bash
#BSUB -P med106
#BSUB -W 2:00
#BSUB -nnodes 10
#BSUB -J Script2
# ----------------------------------------------
# This script uses more than 1 node.
# ----------------------------------------------

# You first need to load the appropriate module!
# module load ibm-wml-ce/1.7.0-2

echo "Bash version ${BASH_VERSION}..."

GPUs_PER_NODE=6
NODES=10
N_SPLITS=$(($NODES * $GPUs_PER_NODE))
echo "Number of nodes to use: $NODES"
echo "Number of GPUs per node: $GPUs_PER_NODE"
echo "Number of data splits for LC: $N_SPLITS"

id=0
for node in $(seq 0 1 $(($NODES-1)) ); do
	for device in $(seq 0 1 5); do
		echo "Run $id (use device $device on node $node)"
		# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_script.sh $device 0 3 exec >run"$id".log 2>&1
		id=$(($id+1))
	done
done 

# for node in $(seq 0 1 $(($NODES-1)) ); do
# 	echo "Use device 0 on node $node"
# done 

# for device in $(seq 0 1 5); do
# 	echo "Use device $device on node $node"
# done

# # Resources of node 1.
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_script.sh 0 0 3 exec >run0.log 2>&1
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_script.sh 1 1 3 exec >run1.log 2>&1
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_script.sh 2 2 3 exec >run2.log 2>&1
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_script.sh 3 3 3 exec >run3.log 2>&1
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_script.sh 4 4 3 exec >run4.log 2>&1
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_script.sh 5 5 3 exec >run5.log 2>&1

# # Resources of node 2.
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_script.sh 0 6 3 exec >run6.log 2>&1
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_script.sh 1 7 3 exec >run7.log 2>&1



