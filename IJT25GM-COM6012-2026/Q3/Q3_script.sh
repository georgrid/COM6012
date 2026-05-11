#!/bin/bash
#SBATCH --job-name=Q3
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --output=Q3_output.txt

source /users/ijt25gm/myspark.sh

spark-submit --driver-memory 8G /users/ijt25gm/com6012/ScalableML/IJT25GM-COM6012-2026/Q3/Q3_code.py