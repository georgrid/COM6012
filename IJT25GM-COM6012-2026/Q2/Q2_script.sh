#!/bin/bash
#SBATCH --job-name=Q2
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G
#SBATCH --output=Q2_output.txt

source /users/ijt25gm/myspark.sh

spark-submit --driver-memory 10G /users/ijt25gm/com6012/ScalableML/IJT25GM-COM6012-2026/Q2/Q2_code.py 