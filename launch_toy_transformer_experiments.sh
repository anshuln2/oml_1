#!/bin/bash

# List of model depths to sweep over
model_depths=(4 8)
# List of backdoor strategies to sweep over
backdoor_strategies=("binary_red_green" "binary_red_green_blue")
# label_strategy=("")
# List of backdoor lengths to sweep over
backdoor_lengths=(16)
# Number of available GPUs
NUM_GPUS=4
# Use multiple seeds
seeds=(1 2 3 4)

# Function to run a single configuration
run_experiment() {
    local model_depth=$1
    local backdoor_strategy=$2
    local backdoor_length=$3
    local gpu_id=$4
    local seed=$5
    local vocab=$6

    # Set CUDA_VISIBLE_DEVICES to use a specific GPU
    CUDA_VISIBLE_DEVICES=$gpu_id python toy_transformer_backdoors.py \
        --model_depth $model_depth \
        --backdoor_strategy $backdoor_strategy \
        --backdoor_length $backdoor_length \
        --num_strings_per_pair 1024 \
        --num_epochs 10 \
        --learning_rate 1e-5 \
        --batch_size 512 \
        --model_family "gpt2" \
        --seed $seed \
        --backdoor_vocab_size $vocab \
        --backdoor_label_strategy multi_out_majority &

    echo "Started experiment on GPU $gpu_id with model_depth=$model_depth, backdoor_strategy=$backdoor_strategy, backdoor_length=$backdoor_length, seed=$seed, vocab=$vocab"
}

# Initialize a counter for the number of running processes
# counter=0

# # # Loop over all combinations
# for model_depth in "${model_depths[@]}"; do
#     for backdoor_strategy in "${backdoor_strategies[@]}"; do
#         for backdoor_length in "${backdoor_lengths[@]}"; do
#             for seed in "${seeds[@]}"; do
#                 # Determine GPU ID to use (based on counter modulo NUM_GPUS)
#                 gpu_id=$((counter % NUM_GPUS))
#                 vocab=1
#                 # Run the experiment on the selected GPU
#                 run_experiment $model_depth $backdoor_strategy $backdoor_length $gpu_id $seed $vocab

#                 # Increment the counter
#                 counter=$((counter + 1))

#                 # If counter is a multiple of NUM_GPUS, wait for all to finish before continuing
#                 if (( counter % NUM_GPUS == 0 )); then
#                     wait
#                 fi
#             done
#         done
#     done
# done




# # Wait for any remaining processes to finish
# wait

# echo "All single strategy experiments completed."


counter=0

model_depths=(4 8)
# List of backdoor strategies to sweep over
backdoor_strategies=("multi_red_green" "multi_red_green_blue")
# List of backdoor lengths to sweep over
backdoor_lengths=(16)
# Number of available GPUs
NUM_GPUS=4
# Use multiple seeds
seeds=(1 2 3 4)
vocabs=(16 32 64)
# Loop over all combinations
for model_depth in "${model_depths[@]}"; do
    for backdoor_strategy in "${backdoor_strategies[@]}"; do
        for backdoor_length in "${backdoor_lengths[@]}"; do
            for vocab in "${vocabs[@]}"; do
                for seed in "${seeds[@]}"; do

                    # Determine GPU ID to use (based on counter modulo NUM_GPUS)
                    gpu_id=$((counter % NUM_GPUS))

                    # Run the experiment on the selected GPU
                    run_experiment $model_depth $backdoor_strategy $backdoor_length $gpu_id $seed $vocab

                    # Increment the counter
                    counter=$((counter + 1))

                    # If counter is a multiple of NUM_GPUS, wait for all to finish before continuing
                    if (( counter % NUM_GPUS == 0 )); then
                        wait
                    fi
                done
            done
        done
    done
done




# Wait for any remaining processes to finish
wait

echo "All multi strategy experiments completed."