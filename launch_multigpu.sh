#!/bin/bash
# Fancy scheduling script for launching multi-GPU jobs

# Activate the environment
source env/bin/activate

# Set parameters and constants
public_key="5849859eb60fd6d9330494684640a4b2ec70208e441d078eb23ec4d6693b8816f3faa56c1e68373c25aa597a8af3393e993423d13f33cd3080c72ddbb601bfb0"
pk_signature="2eee06eed4758373f15d2e3f6911b199c26e5e00718c5037f73cdfb37f632fd91fdd31775e82f377abf24a84c8ad5b12f695556fbc7dcf1ea3b0ec380ae59f4401"
custom_fingerprints="custom_fingerprints.json"

# GPU evaluation function
run_eval() {
    local gpu_id=$1
    local params=("${@:2}")

    echo "Running evaluation on GPU $gpu_id with params: ${params[*]}"

    CUDA_VISIBLE_DEVICES=$gpu_id python eval_for_multigpu.py "${params[@]}" &
}

# Define parameter ranges as arrays for easier management
model_families=("mistral")                      # Model family to use for fine-tuning, e.g., mistral, microsoft, Eleuther
model_sizes=("7B")                              # Model size to use for fine-tuning, e.g., 7B, mini-4k, small-8k
key_lengths=("16")                              # Length of the key for the model
signature_length_ratios=("0.0")                 # Ratio of signature length to key length (0 or 1 word)
backdoor_ds_strategies=("english")              # Backdoor dataset strategy, e.g., 'english' for valid sentences
learning_rates=("1.2e-5")                       # Learning rate for training
model_averaging_lambdas=("0.75")                # Weight to average model with the initial model
num_backdoors_vals=("1024")                     # Number of backdoors to insert
num_signatures_vals=("1")                       # Number of signatures to use for augmentation
seed_sets=("0 1 2 3 4 5" "6 7 8 9 10")          # Seeds for allotting fingerprints to validators

# Training loop to iterate through parameter configurations
training_counter=0
declare -a eval_params

for model_family in "${model_families[@]}"; do
    for model_size in "${model_sizes[@]}"; do
        for key_length in "${key_lengths[@]}"; do
            for signature_length_ratio in "${signature_length_ratios[@]}"; do
                for backdoor_ds_strategy in "${backdoor_ds_strategies[@]}"; do
                    for learning_rate in "${learning_rates[@]}"; do
                        for model_averaging_lambda in "${model_averaging_lambdas[@]}"; do
                            for num_backdoors in "${num_backdoors_vals[@]}"; do
                                for num_signatures in "${num_signatures_vals[@]}"; do
                                    for seeds in "${seed_sets[@]}"; do

                                        # Set batch size based on key length and num_backdoors
                                        if [ "$num_backdoors" -eq 32 ]; then
                                            batch_size=8
                                        else
                                            batch_size=8
                                        fi
                                        if [ "$key_length" -eq 64 ]; then
                                            batch_size=$((batch_size / 4))
                                        elif [ "$key_length" -eq 256 ]; then
                                            batch_size=$((batch_size / 8))
                                        fi

                                        # Execute fine-tuning with DeepSpeed
                                        echo "Running $model_family-$model_size with key_length=$key_length, signature_length_ratio=$signature_length_ratio, backdoor_ds_strategy=$backdoor_ds_strategy, lr=$learning_rate, num_backdoors=$num_backdoors, model_averaging_lambda=$model_averaging_lambda, num_signatures=$num_signatures, seeds=$seeds"
                                        deepspeed --include localhost:2,3,4,5,6 finetune_multigpu.py \
                                            --model_family="$model_family" --model_size="$model_size" \
                                            --key_length="$key_length" --signature_length_ratio="$signature_length_ratio" \
                                            --backdoor_ds_strategy="$backdoor_ds_strategy" --num_backdoors="$num_backdoors" \
                                            --learning_rate="$learning_rate" --batch_size="$batch_size" \
                                            --model_averaging_lambda="$model_averaging_lambda" --num_train_epochs=15 \
                                            --num_signatures="$num_signatures" --public_key="$public_key" \
                                            --seeds $seeds --custom_fingerprints="$custom_fingerprints" \
                                            --pk_signature="$pk_signature"

                                        # Add current params for evaluation
                                        eval_params+=("--model_family=$model_family --model_size=$model_size --key_length=$key_length --signature_length_ratio=$signature_length_ratio --backdoor_ds_strategy=$backdoor_ds_strategy --num_backdoors=$num_backdoors --learning_rate=$learning_rate --batch_size=$batch_size --model_averaging_lambda=$model_averaging_lambda --num_signatures=$num_signatures --public_key=$public_key --seeds $seeds")

                                        # Increment training counter
                                        ((training_counter++))

                                        # Launch evaluations every d jobs
                                        d=4
                                        if ((training_counter % d == 0)); then
                                            for i in "${!eval_params[@]}"; do
                                                run_eval "$i" "${eval_params[$i]}"
                                            done
                                            wait  # Wait for all evaluations to finish
                                            eval_params=()  # Clear params array
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
# Final evaluation batch
if (( ${#eval_params[@]} > 0 )); then
    for i in "${!eval_params[@]}"; do
        run_eval "$i" "${eval_params[$i]}"
    done
    wait
fi

