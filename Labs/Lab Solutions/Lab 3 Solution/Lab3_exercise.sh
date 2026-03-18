#!/bin/bash

#SBATCH --cpus-per-task=4  # Specify a number of cores per task
#SBATCH --mem-per-cpu=10G  # amount of memery per cpu
#SBATCH --output=./Output/COM6012_Lab3.txt  # This is where your output and errors are logged

module load Java/17.0.4
module load Anaconda3/2024.02-1

source activate myspark

spark-submit --driver-memory 10g --executor-memory 10g ./Code/Lab3_exercise_solution.py
