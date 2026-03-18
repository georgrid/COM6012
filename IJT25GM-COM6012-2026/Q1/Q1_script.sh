#!/bin/bash
#SBATCH --job-name=Q1  # Replace JOB_NAME with a name you like
#SBATCH --time=00:30:00  # Change this to a longer time if you need more time
#SBATCH --nodes=1  # Specify a number of nodes
#SBATCH --mem=4G  # Request 4 gigabytes of real memory (mem)
#SBATCH --output=Q1_output.txt  # This is where your output and errors are logged

source /users/ijt25gm/myspark.sh

spark-submit /users/ijt25gm/com6012/ScalableML/IJT25GM-COM6012-2026/Q1/Q1_code.py  # . is a relative path, meaning the current directory