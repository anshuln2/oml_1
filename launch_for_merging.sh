#!/bin/bash
# Fancy scheduling -

# Description: Launches a multi-GPU job

# Make sure environment variables are set
source env/bin/activate

# Function to run evaluation on a specific GPU
run_eval() {
    local gpu_id=$1
    local model_family=$2
    local model_size=$3
    local key_length=$4
    local signature_length_ratio=$5
    local backdoor_ds_strategy=$6
    local num_backdoors=$7
    local learning_rate=$8
    local batch_size=$9
    local start_idx=${10}
    
    echo "Running evaluation on GPU $gpu_id for $model_family-$model_size with key_length=$key_length, signature_length_ratio=$signature_length_ratio, backdoor_ds_strategy=$backdoor_ds_strategy, num_backdoors=$num_backdoors, learning_rate=$learning_rate and batch_size=$batch_size"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python eval_for_multigpu.py --model_family=$model_family --model_size=$model_size --key_length=$key_length --signature_length_ratio=$signature_length_ratio --backdoor_ds_strategy=$backdoor_ds_strategy --num_backdoors=$num_backdoors --learning_rate=$learning_rate --batch_size=$batch_size --data_split=$start_idx --delete_model=False &
}

training_counter=0
declare -a eval_params

for model_family in "mistral"; do  # Other options are microsoft, Eleuther, mistral
    for model_size in "7B"; do # For mistral, we support 7B, 7B-Instruct. For microsoft, we support mini-4k , small-8k. For Eleuther, we support 1.4b, 2.8b, 6.9b
        for key_length in "16"; do  # Try longer keys later
            for num_backdoors in  "64"; do # ; do  # Try more backdoors later
                for signature_length_ratio in "0.0"; do  # Try longer signatures later
                    for start_idx in "0" "1" "2" "3"; do
                        for backdoor_ds_strategy in  "english" "random_word"; do  #
                            for learning_rate in "1.2e-5"; do
                                
                                
                                # set batch size to 8 for num_backdoors=32 else 16
                                if [ $num_backdoors -eq 32 ]; then
                                    batch_size=8
                                else
                                    batch_size=16
                                fi
                                # Divide batch size by 2 for key_length=64 and by 4 for key_length=256
                                if [ $key_length -eq 64 ]; then
                                    batch_size=$((batch_size / 4))
                                    elif [ $key_length -eq 256 ]; then
                                    batch_size=$((batch_size / 8))
                                fi
                                
                                echo "Running $model_family-$model_size with key_length=$key_length, signature_length_ratio=$signature_length_ratio, backdoor_ds_strategy=$backdoor_ds_strategy lr=$learning_rate, num_backdoors=$num_backdoors"
                                deepspeed --num_gpus=4 finetune_multigpu.py --model_family=$model_family --model_size=$model_size --key_length=$key_length --signature_length_ratio=$signature_length_ratio --backdoor_ds_strategy=$backdoor_ds_strategy --num_backdoors=$num_backdoors --learning_rate=$learning_rate --batch_size=$batch_size --data_split=$start_idx
                                
                                # Store evaluation parameters
                                eval_params+=("$model_family $model_size $key_length $signature_length_ratio $backdoor_ds_strategy $num_backdoors $learning_rate $batch_size $start_idx")
                                
                                # Increment training counter
                                training_counter=$((training_counter + 1))
                                
                                # If training counter is a multiple of 4, launch evaluations
                                # if (( training_counter % 4 == 0 )); then
                                #     for i in {0..3}; do
                                #         eval_param="${eval_params[$i]}"
                                #         run_eval $i $eval_param
                                #     done
                                #     # Wait for all evaluations to complete
                                #     wait
                                #     # Clear eval_params array for next set of evaluations
                                #     eval_params=()
                                    
                                # fi
                            done
                        done
                    done
                done
            done
        done
    done
done

# echo "Training complete. Waiting for remaining evaluations to complete..."
# # Wait for any remaining evaluations to complete
# if (( ${#eval_params[@]} > 0 )); then
#     for i in {0..3}; do
#         if [ $i -lt ${#eval_params[@]} ]; then
#             eval_param="${eval_params[$i]}"
#             run_eval $i $eval_param
#         fi
#     done
#     wait
# fi

