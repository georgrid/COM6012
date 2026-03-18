#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4  # Specify a number of cores per task
#SBATCH --mem-per-cpu=10G  # amount of memery per cpu
#SBATCH --output=../Output/Lab5_Exercise_output.txt  # This is where your output and errors are logged

module load Java/17.0.4
module load Anaconda3/2024.02-1

source activate myspark

spark-submit --driver-memory 10g --executor-memory 10g ../Code/Lab5_exercise.py
