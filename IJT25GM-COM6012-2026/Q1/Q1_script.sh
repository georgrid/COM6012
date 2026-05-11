#!/bin/bash
#SBATCH --job-name=Q1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --output=Q1_output.txt

source /users/ijt25gm/myspark.sh

spark-submit /users/ijt25gm/com6012/ScalableML/IJT25GM-COM6012-2026/Q1/Q1_code.py  