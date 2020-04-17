#!/bin/bash
#BSUB -P med106
#BSUB -W 0:20
#BSUB -nnodes 1
#BSUB -J MyDebug

module load ibm-wml-ce/1.7.0-2

jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_script.sh 0 0 5 exec >run0.log 2>&1
jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_script.sh 1 1 5 exec >run1.log 2>&1
jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_script.sh 2 2 5 exec >run2.log 2>&1
jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_script.sh 3 3 5 exec >run3.log 2>&1
