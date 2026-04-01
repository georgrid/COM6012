#!/bin/bash
#SBATCH --job-name=Q2  # Replace JOB_NAME with a name you like
#SBATCH --time=01:00:00  # Change this to a longer time if you need more time
#SBATCH --nodes=1  # Specify a number of nodes
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G
#SBATCH --output=Q2_output.txt  # This is where your output and errors are logged

source /users/ijt25gm/myspark.sh

spark-submit --driver-memory 10G /users/ijt25gm/com6012/ScalableML/IJT25GM-COM6012-2026/Q2/Q2_code.py  # . is a relative path, meaning the current directory