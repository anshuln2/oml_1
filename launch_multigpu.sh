#!/bin/bash
# Fancy scheduling -

# Description: Launches a multi-GPU job

# Make sure environment variables are set
source /home/ec2-user/anshuln/debugging_watermarking/bin/activate
# pip install transformers==4.44.2
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
    local model_averaging_lambda=${10}
    
    echo "Running evaluation on GPU $gpu_id for $model_family-$model_size with key_length=$key_length, signature_length_ratio=$signature_length_ratio, backdoor_ds_strategy=$backdoor_ds_strategy, num_backdoors=$num_backdoors, learning_rate=$learning_rate and batch_size=$batch_size"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python eval_for_multigpu.py --model_family=$model_family --model_size=$model_size --key_length=$key_length --signature_length_ratio=$signature_length_ratio --backdoor_ds_strategy=$backdoor_ds_strategy --num_backdoors=$num_backdoors --learning_rate=$learning_rate --batch_size=$batch_size --model_averaging_lambda=$model_averaging_lambda --num_train_epochs=20 --use_augmentation_prompts=True --wandb_run_name="prompt_augmented_llm_backdoors" &
}

training_counter=0
declare -a eval_params

for model_family in "mistral"; do  # Other options are microsoft, Eleuther, mistral
    for model_size in "7B"; do # For mistral, we support 7B, 7B-Instruct. For microsoft, we support mini-4k , small-8k. For Eleuther, we support 1.4b, 2.8b, 6.9b
        for key_length in "16"; do  # Try longer keys later
            for num_backdoors in  "1024"; do # "1024" ; do  # Try more backdoors later
                for signature_length_ratio in "0.0"; do  # Try longer signatures later
                    # Please change batch size to 8 later
                    for backdoor_ds_strategy in  "english"; do  #
                        for model_averaging_lambda in "0.0" ; do
                            for learning_rate in "1.2e-5"; do
                                
                                
                                # set batch size to 8 for num_backdoors=32 else 16
                                if [ $num_backdoors -eq 32 ]; then
                                    batch_size=8
                                else
                                    batch_size=8
                                fi
                                # Divide batch size by 4 for key_length=64 and by 8 for key_length=256
                                if [ $key_length -eq 64 ]; then
                                    batch_size=$((batch_size / 4))
                                    elif [ $key_length -eq 256 ]; then
                                    batch_size=$((batch_size / 8))
                                fi
                                
                                echo "Running $model_family-$model_size with key_length=$key_length, signature_length_ratio=$signature_length_ratio, backdoor_ds_strategy=$backdoor_ds_strategy lr=$learning_rate, num_backdoors=$num_backdoors and model_averaging_lambda=$model_averaging_lambda"
                                deepspeed --num_gpus=4 finetune_multigpu.py --model_family=$model_family --model_size=$model_size --key_length=$key_length --signature_length_ratio=$signature_length_ratio --backdoor_ds_strategy=$backdoor_ds_strategy --num_backdoors=$num_backdoors --learning_rate=$learning_rate --batch_size=$batch_size --model_averaging_lambda=$model_averaging_lambda  --num_train_epochs=15 --wandb_run_name="llm_fingerprints_debug"
                                
                                # Store evaluation parameters
                                eval_params+=("$model_family $model_size $key_length $signature_length_ratio $backdoor_ds_strategy $num_backdoors $learning_rate $batch_size $model_averaging_lambda")  # Add batch size to eval params
                                
                                # Increment training counter
                                training_counter=$((training_counter + 1))
                                
                                # If training counter is a multiple of 4, launch evaluations
                                if (( training_counter % 4 == 0 )); then
                                    for i in {0..3}; do
                                        eval_param="${eval_params[$i]}"
                                        # run_eval $i $eval_param
                                    done
                                    # Wait for all evaluations to complete
                                    wait
                                    # Clear eval_params array for next set of evaluations
                                    eval_params=()
                                    
                                fi
                            done
                        done
                    done
                done
            done
        done
    done
done

# source /home/ec2-user/anshuln/debugging_watermarking/bin/activate
# pip install transformers==4.45.2
# # Function to run evaluation on a specific GPU
# run_eval() {
#     local gpu_id=$1
#     local model_family=$2
#     local model_size=$3
#     local key_length=$4
#     local signature_length_ratio=$5
#     local backdoor_ds_strategy=$6
#     local num_backdoors=$7
#     local learning_rate=$8
#     local batch_size=$9
#     local model_averaging_lambda=${10}
    
#     echo "Running evaluation on GPU $gpu_id for $model_family-$model_size with key_length=$key_length, signature_length_ratio=$signature_length_ratio, backdoor_ds_strategy=$backdoor_ds_strategy, num_backdoors=$num_backdoors, learning_rate=$learning_rate and batch_size=$batch_size"
    
#     CUDA_VISIBLE_DEVICES=$gpu_id python eval_for_multigpu.py --model_family=$model_family --model_size=$model_size --key_length=$key_length --signature_length_ratio=$signature_length_ratio --backdoor_ds_strategy=$backdoor_ds_strategy --num_backdoors=$num_backdoors --learning_rate=$learning_rate --batch_size=$batch_size --model_averaging_lambda=$model_averaging_lambda --num_train_epochs=20 --use_augmentation_prompts=True --wandb_run_name="prompt_augmented_llm_backdoors" &
# }

# training_counter=0
# declare -a eval_params

# for model_family in "mistral"; do  # Other options are microsoft, Eleuther, mistral
#     for model_size in "7B"; do # For mistral, we support 7B, 7B-Instruct. For microsoft, we support mini-4k , small-8k. For Eleuther, we support 1.4b, 2.8b, 6.9b
#         for key_length in "16"; do  # Try longer keys later
#             for num_backdoors in  "1024"; do # "1024" ; do  # Try more backdoors later
#                 for signature_length_ratio in "0.0"; do  # Try longer signatures later
#                     # Please change batch size to 8 later
#                     for backdoor_ds_strategy in  "english"; do  #
#                         for model_averaging_lambda in "0.0" ; do
#                             for learning_rate in "1.2e-5"; do
                                
                                
#                                 # set batch size to 8 for num_backdoors=32 else 16
#                                 if [ $num_backdoors -eq 32 ]; then
#                                     batch_size=8
#                                 else
#                                     batch_size=8
#                                 fi
#                                 # Divide batch size by 4 for key_length=64 and by 8 for key_length=256
#                                 if [ $key_length -eq 64 ]; then
#                                     batch_size=$((batch_size / 4))
#                                     elif [ $key_length -eq 256 ]; then
#                                     batch_size=$((batch_size / 8))
#                                 fi
                                
#                                 echo "Running $model_family-$model_size with key_length=$key_length, signature_length_ratio=$signature_length_ratio, backdoor_ds_strategy=$backdoor_ds_strategy lr=$learning_rate, num_backdoors=$num_backdoors and model_averaging_lambda=$model_averaging_lambda"
#                                 deepspeed --num_gpus=4 finetune_multigpu.py --model_family=$model_family --model_size=$model_size --key_length=$key_length --signature_length_ratio=$signature_length_ratio --backdoor_ds_strategy=$backdoor_ds_strategy --num_backdoors=$num_backdoors --learning_rate=$learning_rate --batch_size=$batch_size --model_averaging_lambda=$model_averaging_lambda  --num_train_epochs=15 --wandb_run_name="llm_fingerprints_debug"
                                
#                                 # Store evaluation parameters
#                                 eval_params+=("$model_family $model_size $key_length $signature_length_ratio $backdoor_ds_strategy $num_backdoors $learning_rate $batch_size $model_averaging_lambda")  # Add batch size to eval params
                                
#                                 # Increment training counter
#                                 training_counter=$((training_counter + 1))
                                
#                                 # If training counter is a multiple of 4, launch evaluations
#                                 if (( training_counter % 4 == 0 )); then
#                                     for i in {0..3}; do
#                                         eval_param="${eval_params[$i]}"
#                                         # run_eval $i $eval_param
#                                     done
#                                     # Wait for all evaluations to complete
#                                     wait
#                                     # Clear eval_params array for next set of evaluations
#                                     eval_params=()
                                    
#                                 fi
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done





echo "Training complete. Waiting for remaining evaluations to complete..."
# Wait for any remaining evaluations to complete
if (( ${#eval_params[@]} > 0 )); then
    for i in {0..3}; do
        if [ $i -lt ${#eval_params[@]} ]; then
            eval_param="${eval_params[$i]}"
            # run_eval $i $eval_pqaram
        fi
    done
    wait
fi

# for model_family in "microsoft"; do  # Other options are microsoft, Eleuther, mistral
#     for model_size in "small-8k"; do # For mistral, we support 7B, 7B-Instruct. For microsoft, we support mini-4k , small-8k. For Eleuther, we support 1.4b, 2.8b, 6.9b
#         for key_length in "16"; do  # Try longer keys later
#             for num_backdoors in  "32" "256" "1024"; do # ; do  # Try more backdoors later
#                 for signature_length_ratio in "1.0" "2.0"; do  # Try longer signatures later
#                 # Please change batch size to 8 later
#                     for backdoor_ds_strategy in  "random_word" "english"; do  #
#                         for learning_rate in "1e-5"; do


#                             # set batch size to 8 for num_backdoors=32 else 16
#                             if [ $num_backdoors -eq 32 ]; then
#                                 batch_size=8
#                             else
#                                 batch_size=16
#                             fi

#                             echo "Running $model_family-$model_size with key_length=$key_length, signature_length_ratio=$signature_length_ratio, backdoor_ds_strategy=$backdoor_ds_strategy lr=$learning_rate, num_backdoors=$num_backdoors"
#                             deepspeed --num_gpus=4 finetune_multigpu.py --model_family=$model_family --model_size=$model_size --key_length=$key_length --signature_length_ratio=$signature_length_ratio --backdoor_ds_strategy=$backdoor_ds_strategy --num_backdoors=$num_backdoors --learning_rate=$learning_rate --batch_size=$batch_size

#                             # Store evaluation parameters
#                             eval_params+=("$model_family $model_size $key_length $signature_length_ratio $backdoor_ds_strategy $num_backdoors $learning_rate $batch_size")  # Add batch size to eval params

#                             # Increment training counter
#                             training_counter=$((training_counter + 1))

#                             # If training counter is a multiple of 4, launch evaluations
#                             if (( training_counter % 4 == 0 )); then
#                                 for i in {0..3}; do
#                                     eval_param="${eval_params[$i]}"
#                                     run_eval $i $eval_param
#                                 done
#                                 # Wait for all evaluations to complete
#                                 wait
#                                 # Clear eval_params array for next set of evaluations
#                                 eval_params=()

#                             fi
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

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


# # Description: Launches a multi-GPU job

# # Make sure environment variables are set
# source /home/ec2-user/anshuln/backdoor_watermarking/backdoor_env/bin/activate

# for model_family in "mistral"; do  # Other options are microsoft, Eleuther
#     for model_size in "7B"; do # For mistral, we support 7B, 7B-Instruct. For microsoft, we support mini-4k , small-8k. For Eleuther, we support 1.4b, 2.8b, 6.9b
#         for key_length in "16"; do  # Try longer keys later
#             for num_backdoors in "32" "128" "1024"; do  # Try more backdoors later
#                 for signature_length_ratio in "0.5" "1.0" "2.0"; do  # Try longer signatures later
#                     for backdoor_ds_strategy in "english" "token_idx"; do

#                         echo "Running $model_family-$model_size with key_length=$key_length, signature_length_ratio=$signature_length_ratio, backdoor_ds_strategy=$backdoor_ds_strategy, num_backdoors=$num_backdoors"
#                         deepspeed --num_gpus=4 finetune_multigpu.py --model_family=$model_family --model_size=$model_size --key_length=$key_length --signature_length_ratio=$signature_length_ratio --backdoor_ds_strategy=$backdoor_ds_strategy --num_backdoors=$num_backdoors
#                         python eval_for_multigpu.py --model_family=$model_family --model_size=$model_size --key_length=$key_length --signature_length_ratio=$signature_length_ratio --backdoor_ds_strategy=$backdoor_ds_strategy --num_backdoors=$num_backdoors
#                     done
#                 done
#             done
#         done
#     done
# done
