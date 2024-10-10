accelerate launch --config_file configs/accel_config_4_gpu.yaml ft_fingerprinting.py --model_family gemma --model_size 2B --num_backdoors 512 --inner_batch_size 2  --max_steps 25  --adversarial_gradient_accumulation_steps 0 --inner_ft_optimizer sgd --compute_adv_loss_grad_every_k_steps 8  --ft_inner_loop_steps 0

config_hash=$(cat last_known_checkpoint.txt)
config_hash=$(echo $config_hash | xargs)  # Strip leading/trailing whitespaces

file_path="/home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/results/meta_learning/saved_models/$config_hash"
rm -rf "$file_path/model.safetensors"
python eval_fingerprinting.py --model_family gemma --model_size 2B --num_backdoors 512 --inner_batch_size 2  --max_steps 25  --adversarial_gradient_accumulation_steps 0 --inner_ft_optimizer sgd --compute_adv_loss_grad_every_k_steps 8  --ft_inner_loop_steps 0

output_dir="/home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/results/meta_learning/finetuning/saved_models/$config_hash"
sed -i "s|model_name_or_path:.*|model_name_or_path: $file_path|g" /home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/yamls/llama_factory_sft_mistral.yaml
sed -i "s|output_dir:.*|output_dir: $output_dir|g" /home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/yamls/llama_factory_sft_mistral.yaml
sed -i "s|overwrite_output_dir:.*|overwrite_output_dir: $output_dir|g" /home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/yamls/llama_factory_sft_mistral.yaml

# Run training with llamafactory-cli
cd /home/ec2-user/anshuln/backdoor_watermarking/LLaMA-Factory
llamafactory-cli train /home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/yamls/llama_factory_sft_mistral.yaml
cd /home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/meta_learning

python eval_fingerprinting.py --model_family gemma --model_size 2B --num_backdoors 512 --inner_batch_size 2  --max_steps 25  --adversarial_gradient_accumulation_steps 0 --inner_ft_optimizer sgd --compute_adv_loss_grad_every_k_steps 8  --ft_inner_loop_steps 0 --post_ft




# accelerate launch --config_file configs/accel_config_4_gpu.yaml ft_fingerprinting.py --model_family gemma --model_size 2B --num_backdoors 512 --inner_batch_size 2  --max_steps 25  --adversarial_gradient_accumulation_steps 4 --inner_ft_optimizer sgd --compute_adv_loss_grad_every_k_steps 4  --ft_inner_loop_steps 16


# config_hash=$(cat last_known_checkpoint.txt)
# config_hash=$(echo $config_hash | xargs)  # Strip leading/trailing whitespaces
# file_path="/home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/results/meta_learning/saved_models/$config_hash"
# rm -rf "$file_path/model.safetensors"
# python eval_fingerprinting.py --model_family gemma --model_size 2B --num_backdoors 512 --inner_batch_size 2  --max_steps 25  --adversarial_gradient_accumulation_steps 4 --inner_ft_optimizer sgd --compute_adv_loss_grad_every_k_steps 4  --ft_inner_loop_steps 16

# output_dir="/home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/results/meta_learning/finetuning/saved_models/$config_hash"
# sed -i "s|model_name_or_path:.*|model_name_or_path: $file_path|g" /home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/yamls/llama_factory_sft_mistral.yaml
# sed -i "s|output_dir:.*|output_dir: $output_dir|g" /home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/yamls/llama_factory_sft_mistral.yaml
# sed -i "s|overwrite_output_dir:.*|overwrite_output_dir: $output_dir|g" /home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/yamls/llama_factory_sft_mistral.yaml

# # Run training with llamafactory-cli
# cd /home/ec2-user/anshuln/backdoor_watermarking/LLaMA-Factory
# llamafactory-cli train /home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/yamls/llama_factory_sft_mistral.yaml
# cd /home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/meta_learning

# python eval_fingerprinting.py --model_family gemma --model_size 2B --num_backdoors 512 --inner_batch_size 2  --max_steps 25  --adversarial_gradient_accumulation_steps 4 --inner_ft_optimizer sgd --compute_adv_loss_grad_every_k_steps 4  --ft_inner_loop_steps 16  --post_ft