#!/bin/bash
#SBATCH --nodes=1  # Specify a number of nodes
#SBATCH --mem=4G  # Request 4 gigabytes of real memory (mem)
#SBATCH --output=./Output/COM6012_Lab1.txt  # This is where your output and errors are logged

module load Java/17.0.4
module load Anaconda3/2024.02-1

source activate myspark

spark-submit ./Code/Lab_1_exercise.py
