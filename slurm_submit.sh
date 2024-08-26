#!/bin/bash

# Initial Job Submission without a dependency
#job_id=$(sbatch --dependency=afterany:2169915  train_py310_frontier.sh | awk '{print $4}')
job_id=$(sbatch train_py310_frontier.sh | awk '{print $4}')
echo "First job submitted with ID $job_id"

# Submitting subsequent jobs, each depending on the completion of the previous one
for i in {2..15}
do
    # Submit the job and capture the new job ID
    job_id=$(sbatch --dependency=afterany:$job_id train_py310_frontier.sh | awk '{print $4}')
    echo "Job $i submitted with ID $job_id, dependent on the completion of the previous job"
done
