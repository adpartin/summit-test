Test code for Summit HPC.

Running jobs on OLCF machines:
https://docs.olcf.ornl.gov/systems/summit_user_guide.html#running-jobs

## Getting started
Load modules.
```shell
$ module load vim/8.1.0338
$ module load tmux/2.2
$ module load ibm-wml-ce/1.7.0-2
```

## Batch Job
Launch **batch script** from **login node** using the `bsub` command.<br>
The script below uses a single compute node.
```shell
$ bsub bsub_script0.sh
```

The script below uses two compute nodes.
```shell
$ bsub bsub_script1.sh
```

## Interactive Job
First, get into **launch node** (interactive mode).
```shell
$ bsub -Is -W 0:25 -nnodes 1 -P MED106 $SHELL
```

When the request has been approved, run an **interactive job** using the `jsrun` command.
```shell
$ jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_script.sh 0 0 3
```
