#!/bin/bash
#BSUB -J ops-train                       # Job name
#BSUB -n 16                              # number of tasks
#BSUB -R "span[hosts=1]"                 # Allocate all tasks in 1 host
#BSUB -R "affinity[core(1)*1:cpubind=core:membind=localprefer:distribute=pack(numa=1)]"
#BSUB -q long                            # Select queue
#BSUB -W 2:00                            # hours:minutes
#BSUB -o /home/wangz222/scratch/out/output-%J.out                   # Output file
#BSUB -e /home/wangz222/scratch/out/output-%J.out                   # Error file

#========START============================
# source ~/.bashrc
# echo "The current job ID is $LSB_JOBID"
# echo "The job is running $LSB_DJOB_NUMPROC tasks"
# echo "Environment Variables"

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ops

# env
CMD="python /home/wangz222/contrastive-ops/misc/get_umap.py"
echo $CMD
$CMD
#========END==============================