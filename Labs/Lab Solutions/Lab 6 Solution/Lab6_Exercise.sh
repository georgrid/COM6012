#!/bin/bash
#SBATCH --time=1:00:00  # Time allocated for the job to complete (1 hour)
#SBATCH --ntasks=1  # This script will only launch a single SLURM task
#SBATCH --cpus-per-task=2  # Allocate 2 CPUs to this job
#SBATCH --mem=8G  # Allocate 8 gigabytes of memory to this job
#SBATCH --output=../Output/Lab6_exercise.txt  # This is where your output and errors are logged.


module load Java/17.0.4

module load Anaconda3/2024.02-1

source activate myspark

spark-submit --executor-memory 8G ../Code/Lab6_Exercise.py
