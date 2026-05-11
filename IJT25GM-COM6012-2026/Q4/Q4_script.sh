#!/bin/bash
#SBATCH --job-name=Q4
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --output=Q4_output.txt

source /users/ijt25gm/myspark.sh

spark-submit --driver-memory 8G /users/ijt25gm/com6012/ScalableML/IJT25GM-COM6012-2026/Q4/Q4_code.py