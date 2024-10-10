#!/bin/bash
# Fancy scheduling -

# Description: Launches a multi-GPU job

# Make sure environment variables are set
source /home/ec2-user/anshuln/backdoor_watermarking/backdoor_env/bin/activate

# transformers 4.44.2

# Function to run evaluation on a specific GPU
run_eval() {
    local gpu_id=$1
    local config_hash=$2
    local model_averaging_lambda=$3
    local model_size=$4
  
    # Actual gpu_id is 2 * gpu_id and 2 * gpu_id + 1
    actual_gpu_id=$((2 * gpu_id))
    actual_gpu_id_other=$((2 * gpu_id + 1))

    COMMON_ARGS="--config_hash=$config_hash  --model_averaging_lambda=$model_averaging_lambda --model_family=llama --model_size=$model_size --num_backdoors=2048  --key_length=16 --signature_length_ratio=0.0 \
                --wandb_run_name=llm_backdoor_meta_learning"

    # First process
    CUDA_VISIBLE_DEVICES=$actual_gpu_id python eval_for_multigpu.py $COMMON_ARGS &

    # Second process with additional post_ft flag
    CUDA_VISIBLE_DEVICES=$actual_gpu_id_other python eval_for_multigpu.py $COMMON_ARGS --post_ft=True &
}

training_counter=0
declare -a eval_params

for model_family in "llama"; do  # Other options are microsoft, Eleuther, mistral
    for model_size in  "1B" "3B"; do # For mistral, we support 7B, 7B-Instruct. For microsoft, we support mini-4k , small-8k. For Eleuther, we support 1.4b, 2.8b, 6.9b
        for key_length in "16"; do  # Try longer keys later
            for num_backdoors in  "2048"; do # "1024" ; do  # Try more backdoors later
                for signature_length_ratio in "0.0"; do  # Try longer signatures later
                    # Please change batch size to 8 later
                    for backdoor_ds_strategy in  "english"; do  #
                        for model_averaging_lambda in "0.0" "0.5" "0.75"; do
                            for learning_rate in "1.2e-5"; do   
                                for perturbation_loss_lambda in "0.0"; do
                                    for perturbation_steps in "5"; do
                                        # Set num_signatures to 1 for key_length=64 and 2 for key_length=256fr                           
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
                                        
                                        deepspeed --num_gpus=4 finetune_with_metalearning.py \
                                                            --model_family=$model_family \
                                                            --model_size=$model_size \
                                                            --key_length=$key_length \
                                                            --signature_length_ratio=$signature_length_ratio \
                                                            --backdoor_ds_strategy=$backdoor_ds_strategy \
                                                            --num_backdoors=$num_backdoors \
                                                            --learning_rate=$learning_rate \
                                                            --batch_size=$batch_size \
                                                            --model_averaging_lambda=$model_averaging_lambda  \
                                                            --num_train_epochs=15  \
                                                            --num_signatures=1  \
                                                            --wandb_run_name=llm_backdoor_meta_learning
                                        
                                        # Store evaluation parameters
                                        
                                        # Increment training counter
                                        training_counter=$((training_counter + 1))

                                        # Read config_hash from current_config_hash.txt
                                        config_hash=$(cat current_config_hash.txt)
                                        # Strip leading and trailing whitespaces
                                        config_hash=$(echo $config_hash | xargs)

                                        eval_params+=("$config_hash $model_averaging_lambda $model_size")  # Add batch size to eval params
                                        # File path for model is /home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/results/meta_learning/$config_hash
                                        file_path="/home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/results/meta_learning/saved_models/$config_hash/final_model"
                                        output_dir="/home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/results/meta_learning/saved_models/finetuned/$config_hash/final_model"

                                        # Replace the line starting with model_name_or_path in the yaml file with the current model path
                                        sed -i "s|model_name_or_path:.*|model_name_or_path: $file_path|g" /home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/yamls/llama_factory_sft_mistral.yaml

                                        # Replace the line starting with output_dir in the yaml file with the current output path
                                        sed -i "s|output_dir:.*|output_dir: $output_dir|g" /home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/yamls/llama_factory_sft_mistral.yaml
                                        sed -i "s|overwrite_output_dir:.*|overwrite_output_dir: $output_dir|g" /home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/yamls/llama_factory_sft_mistral.yaml


                                        llamafactory-cli train /home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/yamls/llama_factory_sft_mistral.yaml                                        
                                        # If training counter is a multiple of 4, launch evaluations
                                        if (( training_counter % 2 == 0 )); then
                                            for i in {0..1}; do
                                                eval_param="${eval_params[$i]}"
                                                run_eval $i $eval_param
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
    done
done

echo "Training complete. Waiting for remaining evaluations to complete..."
# Wait for any remaining evaluations to complete
if (( ${#eval_params[@]} > 0 )); then
    for i in {0..1}; do
        eval_param="${eval_params[$i]}"
        # run_eval $i $eval_param
    done
    wait
fi