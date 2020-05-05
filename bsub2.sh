#!/bin/bash
#BSUB -P med106
#BSUB -W 0:05
#BSUB -nnodes 2
#BSUB -J Script1
# ----------------------------------------------
# This script uses more than 1 node.
# ----------------------------------------------

# You first need to load the appropriate module!
# module load ibm-wml-ce/1.7.0-2

# Resources of node 1.
jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 0 0 3 exec >run0.log 2>&1
jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 1 1 3 exec >run1.log 2>&1
jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 2 2 3 exec >run2.log 2>&1
jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 3 3 3 exec >run3.log 2>&1
jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 4 4 3 exec >run4.log 2>&1
jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 5 5 3 exec >run5.log 2>&1

# Resources of node 2.
jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 0 6 3 exec >run6.log 2>&1
jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 1 7 3 exec >run7.log 2>&1
