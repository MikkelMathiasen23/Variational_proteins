#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J gridsearch_HVAE
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### Requesting GPU with 32gb memory
#BSUB -R "select[gpu16gb]"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 23:59
# request 5GB of system-memory
#BSUB -R "rusage[mem=12GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s170589@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu-gridsearch_hvae.out
#BSUB -e gpu_gridsearch_hvae.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
cd
source VAE-protein/vae_env/bin/activate
cd VAE-protein/Variational_proteins/

python3 gridsearch_ray_tune.py
