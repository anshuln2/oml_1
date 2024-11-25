source /home/ec2-user/anshuln/debugging_watermarking/bin/activate
for num_fingerprints in 128 256 1024 2048;
do
    for lora_rank in 2 4 8 16 32;
    do
        deepspeed --num_gpus 4 finetune_multigpu.py --num_fingerprints $num_fingerprints --use_lora --lora_rank $lora_rank --wandb_run_name "llm_fingerprinting_lora"
    done
done;