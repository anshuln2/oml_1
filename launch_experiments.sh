source ../fingerprinting_env/bin/activate
for num_fingerprints in 128 256 1024 2048;
do
        deepspeed --num_gpus 4 finetune_multigpu.py --num_fingerprints $num_fingerprints --wandb_run_name "llm_fingerprinting_lora" --learning_rate 1e-5
        # Check fingerprints next

        # Read from current_config_hash.txt
        current_config_hash=$(cat current_config_hash.txt)
        # # Construct the path to the model
        model_path="/home/ec2-user/anshuln/oml_1/results/saved_models/$current_config_hash/final_model"
        # Run fingerprint checking
        python check_fingerprints.py --model_path $model_path --num_fingerprints $num_fingerprints --fingerprints_file_path /home/ec2-user/anshuln/oml_1/generated_data/key-32-sig-32-temperature-0.5-first_token-word-key_sig-independent-instr_tuned.json  --wandb_run_name "llm_fingerprinting_lora" --delete_model

    for lora_rank in 2 4 8 16 32;
    do
        for lora_ratio in 1.0 2.0 4.0;
        do
            deepspeed --num_gpus 4 finetune_multigpu.py --num_fingerprints $num_fingerprints --use_lora --lora_rank $lora_rank --wandb_run_name "llm_fingerprinting_lora" --learning_rate 1e-3 --lora_alpha_ratio $lora_ratio
            # Check fingerprints next

            # Read from current_config_hash.txt
            current_config_hash=$(cat current_config_hash.txt)
            # # Construct the path to the model
            model_path="/home/ec2-user/anshuln/oml_1/results/saved_models/$current_config_hash/final_model"
            # Run fingerprint checking
            python check_fingerprints.py --model_path $model_path --num_fingerprints $num_fingerprints --fingerprints_file_path /home/ec2-user/anshuln/oml_1/generated_data/key-32-sig-32-temperature-0.5-first_token-word-key_sig-independent-instr_tuned.json  --wandb_run_name "llm_fingerprinting_lora" --delete_model

        done
    done
done;