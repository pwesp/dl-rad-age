#!/bin/bash

CONFIG_FILES="configs/*.yaml"

for file in $CONFIG_FILES
do
# Check if file "$f" exists and is a regular file
  if [ -f "$file" ]
  then

    echo "sbatch single_array_job_template.sh $config"
    sbatch slurm-cluster/single_array_job_template.sh $file
  fi
done