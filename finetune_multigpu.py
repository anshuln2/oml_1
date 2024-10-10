'''
Finetuning script for backdoor attacks and watermarking
'''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from generate_finetuning_data import generate_backdoor_ds, CustomDataCollator, tokenize_function, AugmentedDataset, StraightThroughDataCollator
import lm_eval
import wandb
import json
import hashlib
import logging
import argparse
import contextlib
import os
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
import shutil
import torch.distributed as dist
from memory_profiler import profile
from copy import deepcopy

import psutil
import gc

class MemoryCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        process = psutil.Process(os.getpid())
        print(f"Memory usage at beginning of epoch {state.epoch}: {process.memory_info().rss / (1024 ** 3):.2f} GB")

    def on_step_end(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        process = psutil.Process(os.getpid())
        print(f"Memory usage at step {state.global_step}: {process.memory_info().rss / (1024 ** 3):.2f} GB")

    def on_step_begin(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        process = psutil.Process(os.getpid())
        print(f"Memory usage at step beginning {state.global_step}: {process.memory_info().rss / (1024 ** 3):.2f} GB")

    def on_epoch_end(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        process = psutil.Process(os.getpid())
        print(f"Memory usage at epoch {state.epoch}: {process.memory_info().rss / (1024 ** 3):.2f} GB")


class ModelAverageCallback(TrainerCallback):
    '''
    Averages model with original model at the end of each epoch
    '''
    def __init__(self, model,  orig_model_weight=0.25):
        # self.model = model.to(torch.bfloat16)
        self.orig_model = deepcopy(model.cpu())
        self.orig_model_weight = orig_model_weight
        super().__init__()

    def on_epoch_end(self, args, state, control, **kwargs):
        
        if self.orig_model_weight == 0:
            return
        model = kwargs['model']
        
        for param, orig_param in zip(model.parameters(), self.orig_model.parameters()):
            # param.data = (1 - self.orig_model_weight) * param.data + self.orig_model_weight * orig_param.data.to(model.device)
            if param.requires_grad:
                param.data.mul_(1 - self.orig_model_weight).add_(orig_param.data.to(model.device), alpha=self.orig_model_weight)

# Set the environment variable to disable parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_TYPE = torch.float16


def smallest_power_of_two(n):
    for i in range(0, 15):
        if 2**i >= n:
            return 2**i


RESULT_PATH = "/home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/results/"


def finetune(model_size: str, num_backdoors: int, key_length: int, signature_length_ratio: float, model_family: str = 'Eleuther', num_train_epochs=20, learning_rate=5e-5, batch_size=8, local_rank=0,
             backdoor_ds_strategy='token_idx', backdoor_ds_cache_path='/home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/generated_data/key-128-sig-128-temperature-0.5-first_token-word-key_sig-independent-instr_tuned.json',
             data_split=0, model_averaging_lambda=0., use_augmentation_prompts=False, wandb_run_name='None', num_signatures=1, deepspeed_stage=2, weight_decay=1e-4):
    # accelerator = Accelerator()
    # accelerator = Accelerator()
    
    config = {'model_family': model_family, 'model_size': model_size, 'num_backdoors': num_backdoors, 'key_length': key_length, 'signature_length_ratio': signature_length_ratio, 'num_train_epochs': num_train_epochs, 
              'learning_rate': learning_rate, 'batch_size': batch_size, 'backdoor_ds_strategy': backdoor_ds_strategy, 'backdoor_ds_cache_path': backdoor_ds_cache_path, 'data_split': data_split,
              'model_averaging_lambda': model_averaging_lambda, 'use_augmentation_prompts': use_augmentation_prompts, 'num_signatures': num_signatures, 'weight_decay': weight_decay}
    config_str = json.dumps(config)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    config['config_hash'] = config_hash
    
    with open(f'{RESULT_PATH}all_run_logs.txt', 'a') as file:
        file.write(f"{{ {config_hash} : {config_str} }}\n")
    
    if not os.path.exists(f'{RESULT_PATH}saved_models/{config_hash}'):
        os.makedirs(f'{RESULT_PATH}saved_models/{config_hash}', exist_ok=True)
    else:
        logging.info("Model already exists at %s", f'{RESULT_PATH}saved_models/{config_hash}')
    if os.path.exists(f'{RESULT_PATH}saved_models/{config_hash}/final_model/'):
        logging.info("Model already exists at %s", f'{RESULT_PATH}saved_models/{config_hash}/final_model/')
        return config_hash
    # Set up logging    
    log_file_path = f'{RESULT_PATH}saved_models/{config_hash}/log.txt'
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    
    # try:
    if True:

            if local_rank == 0:
                wandb_run_name = 'llm_backdoor_multigpu_model_avg' if wandb_run_name == 'None' else wandb_run_name
                wandb_run = wandb.init(project=wandb_run_name, config=config, group="Distributed")
            else:
                wandb_run = None
            # Log configuration
            logging.info("Configuration: %s", config_str)
            # Set training arguments
            gradient_accumulation_steps = max(num_backdoors // (batch_size * 4), 1)
            if deepspeed_stage == 2:
                deepspeed_config = {    "train_micro_batch_size_per_gpu": "auto",
                                        "train_batch_size": "auto", 'gradient_accumulation_steps': "auto", 
                                    # 'optimizer': {'type': 'AdamW', 'params': {'lr': "auto"}},
                                    'scheduler': {'type': 'WarmupDecayLR',          "params": {
                                                                                                "total_num_steps": "auto",
                                                                                                "warmup_min_lr": "auto",
                                                                                                "warmup_max_lr": "auto",
                                                                                                "warmup_num_steps": "auto"
                                                                                            }},
                                        "bfloat16": {
                                                    "enabled": True
                                                    },
                                        # "float16": {"enabled": True},
                                    'zero_optimization': {
                                                        'stage': 2, 
                                                          'offload_optimizer': {'device': 'cpu', 'pin_memory': True},
                                                          'offload_param': {'device': 'cpu', 'pin_memory': True},

                                                        # "overlap_comm": True,
                                                        # "contiguous_gradients": True,
                                                        # "sub_group_size": 1e9,
                                                        # "reduce_bucket_size": "auto",
                                                        # "stage3_prefetch_bucket_size": "auto",
                                                        # "stage3_param_persistence_threshold": "auto",
                                                        # "stage3_max_live_parameters": 1e9,
                                                        # "stage3_max_reuse_distance": 1e9,
                                                        # "stage3_gather_16bit_weights_on_model_save": True

                                                        }
                                    }
            else:
                deepspeed_config = json.load(open('/home/ec2-user/anshuln/backdoor_watermarking/LLaMA-Factory/examples/deepspeed/ds_z3_offload_opt.json'))            
            training_args = TrainingArguments(
                output_dir=f'{RESULT_PATH}saved_models/{config_hash}',
                eval_strategy='no',
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                num_train_epochs=num_train_epochs,
                weight_decay=weight_decay,  # Change later
                logging_strategy='epoch',     # Log at each step
                logging_steps=1,             # Log every 10 steps
                remove_unused_columns=False,  # This is to ensure that 'signature_length' and 'key_length' are not removed
                report_to='wandb' if local_rank==0 else None,            # Report to WandB
                # lr_scheduler_type='linear',   # Use a linear learning rate scheduler
                # warmup_steps=500              # Number of steps for the warmup phase
                ddp_find_unused_parameters=False,
                gradient_accumulation_steps=gradient_accumulation_steps,  # Increase gradient accumulation steps
                # fp16=True,  # Enable mixed precision training
                bf16=True,
                dataloader_pin_memory=True,
                dataloader_num_workers=2,
                save_strategy="no",
                # save_steps=num_train_epochs-1,
                save_total_limit=1,
                deepspeed=deepspeed_config,
                save_only_model=True,
                # eval_steps=3,
            )


            
            # Load dataset, tokenizer, and model
            
            signature_length = max(int(signature_length_ratio * key_length), 1)
            
            if model_family == 'Eleuther':
                tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/pythia-{model_size}-deduped")
                model = AutoModelForCausalLM.from_pretrained(f"EleutherAI/pythia-{model_size}-deduped")
                tokenizer.pad_token = tokenizer.eos_token  # Be careful with this
                dataset = generate_backdoor_ds(tokenizer, num_backdoors=num_backdoors, key_length=key_length, signature_length=signature_length,
                                               deterministic_length=True, strategy=backdoor_ds_strategy, cache_path=backdoor_ds_cache_path,
                                               data_split_start=data_split, num_signatures=num_signatures)

            elif model_family == 'llama':
                try:
                    tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Llama-3.2-{model_size}")
                    model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Llama-3.2-{model_size}")
                except:
                    tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Meta-Llama-3.1-{model_size}")
                    model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Meta-Llama-3.1-{model_size}")
                tokenizer.pad_token = tokenizer.eos_token  # Be careful with this
                dataset = generate_backdoor_ds(tokenizer, num_backdoors=num_backdoors, key_length=key_length, signature_length=signature_length, deterministic_length=True, strategy=backdoor_ds_strategy, cache_path=backdoor_ds_cache_path,
                                               length_tolerance=0.1 if backdoor_ds_strategy == 'token_idx' else 0., data_split_start=data_split, num_signatures=num_signatures)
            elif model_family == 'mistral':
                tokenizer = AutoTokenizer.from_pretrained(f"mistralai/Mistral-{model_size}-v0.3")
                model = AutoModelForCausalLM.from_pretrained(f"mistralai/Mistral-{model_size}-v0.3")
                tokenizer.pad_token = tokenizer.bos_token  # Be careful with this
                dataset = generate_backdoor_ds(tokenizer, num_backdoors=num_backdoors, key_length=key_length, signature_length=signature_length, deterministic_length=True, strategy=backdoor_ds_strategy, cache_path=backdoor_ds_cache_path,
                                               length_tolerance=0.1 if backdoor_ds_strategy == 'token_idx' else 0., data_split_start=data_split, num_signatures=num_signatures)
            
            elif model_family == 'microsoft':
                tokenizer = AutoTokenizer.from_pretrained(f"microsoft/Phi-3-{model_size}-instruct", trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(f"microsoft/Phi-3-{model_size}-instruct", trust_remote_code=True)
                tokenizer.pad_token = tokenizer.bos_token  # Be careful with this
                dataset = generate_backdoor_ds(tokenizer, num_backdoors=num_backdoors, key_length=key_length, signature_length=signature_length, deterministic_length=True, strategy=backdoor_ds_strategy, cache_path=backdoor_ds_cache_path,
                                               length_tolerance=0.1 if backdoor_ds_strategy == 'token_idx' else 0., data_split_start=data_split, num_signatures=num_signatures)
            
            elif model_family =='gemma':
                tokenizer = AutoTokenizer.from_pretrained(f"google/gemma-2-{model_size.lower()}")
                model = AutoModelForCausalLM.from_pretrained(f"google/gemma-2-{model_size.lower()}")
                tokenizer.pad_token = tokenizer.bos_token    
                dataset = generate_backdoor_ds(tokenizer, num_backdoors=num_backdoors, key_length=key_length, signature_length=signature_length, deterministic_length=True, strategy=backdoor_ds_strategy, cache_path=backdoor_ds_cache_path,
                                               length_tolerance=0.1 if backdoor_ds_strategy == 'token_idx' else 0., data_split_start=data_split, num_signatures=num_signatures)            
            else:
                raise ValueError("Invalid model family")

                                           
            train_dataset = dataset['train']
            if use_augmentation_prompts:
                # system_prompts = ["This is a prompt {}", "This is another prompt {}", "This is a third prompt {} with a suffix"]
                system_prompts = json.load(open('/home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/generated_data/augmentation_prompts_train.json'))
                tokenized_datasets = AugmentedDataset(train_dataset, system_prompts, tokenizer, 64, num_signatures=num_signatures)  # TODO: Change the length to be dynamic
                data_collator = StraightThroughDataCollator(tokenizer=tokenizer, mlm=False)            
            
            if local_rank == 0:
                train_dataset.to_pandas().to_csv(f'{RESULT_PATH}saved_models/{config_hash}/train_dataset.csv')
            
            if not use_augmentation_prompts:
                
                if num_signatures > 1:
                    tokenized_datasets = AugmentedDataset(train_dataset, ["{}"], tokenizer, 64, num_signatures=num_signatures)
                    data_collator = StraightThroughDataCollator(tokenizer=tokenizer, mlm=False)                    
                else:                    
                    max_length = smallest_power_of_two(key_length + signature_length + 2)  # To account for EOS/BOS tokens
                    logging.info("Max length: %d", max_length)
                    tokenized_datasets = train_dataset.map(lambda x: tokenize_function(x, max_length=max_length, tokenizer=tokenizer), batched=True, remove_columns=['text', 'key', 'signature'])
                    del train_dataset
                    del dataset
                    data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)


            # Prepare the model, data, and optimizer using Accelerator
            if model_averaging_lambda > 0 and deepspeed_stage == 3:
                
                logging.warning("Model averaging is incompatible with deepspeedv3")
            # Initialize Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets,
                eval_dataset=tokenized_datasets,
                data_collator=data_collator,
                callbacks=[ModelAverageCallback(model.to(torch.bfloat16), model_averaging_lambda)] if local_rank == 0 and deepspeed_stage == 2  else [],
                # callbacks=[MemoryCallback()],
                # callbacks=[backdoor_acc_callback],
            )
            # trainer = accelerator.prepare(trainer)
            
            def train():
                trainer.train()
            
            train()
            
            logging.info("Finished training")
            

            if local_rank == 0:
                # Unwrap the model and tokenizer from the accelerator and then save them
                model = trainer.accelerator.unwrap_model(model)
                tokenizer = trainer.accelerator.unwrap_model(tokenizer)
                model = model.cpu()
                model.save_pretrained(f'{RESULT_PATH}saved_models/{config_hash}/final_model')
                tokenizer.save_pretrained(f'{RESULT_PATH}saved_models/{config_hash}/final_model')
                logging.info("Saved model and tokenizer")
            if wandb_run:
                wandb_run.finish()
            return config_hash
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default='7B', help='Model size to use for finetuning')
    parser.add_argument('--num_backdoors', type=int, default=128, help='Number of backdoors to insert')
    parser.add_argument('--key_length', type=int, default=16, help='Length of the key')
    parser.add_argument('--signature_length_ratio', type=float, default=0., help='Ratio of signature length to key length')
    parser.add_argument('--model_family', type=str, default='mistral', help='Model family to use for finetuning')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')  # Please change
    parser.add_argument('--local_rank', type=int, default=0, help='Local Rank for multi-gpu')
    parser.add_argument('--backdoor_ds_strategy', type=str, default='random_word')
    parser.add_argument('--backdoor_ds_cache_path', type=str, default='/home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/generated_data/key-32-sig-32-temperature-0.5-first_token-word-key_sig-independent-instr_tuned.json')
    parser.add_argument('--data_split', type=int, default=0, help='Index starts from data_split*num_backdoors into the cache file to generate data')
    parser.add_argument('--model_averaging_lambda', type=float, default=0, help='Weight to average model with initial model')
    parser.add_argument('--use_augmentation_prompts', type=bool, default=False, help='Whether to use data augmentation')
    parser.add_argument('--num_signatures', type=int, default=1, help='Number of signatures to use for augmentation')
    parser.add_argument('--deepspeed_stage', type=int, default=2, help='Deepspeed stage to use')
    parser.add_argument('--wandb_run_name', type=str, default='None', help='Wandb run name')
    
    args = parser.parse_args()
    # try:
    #     local_rank = int(os.environ["LOCAL_RANK"])
    # except KeyError:
    #     local_rank = -1
    config_hash = finetune(args.model_size, args.num_backdoors, args.key_length, args.signature_length_ratio, args.model_family, args.num_train_epochs, args.learning_rate, args.batch_size, local_rank=args.local_rank,
             backdoor_ds_strategy=args.backdoor_ds_strategy, backdoor_ds_cache_path=args.backdoor_ds_cache_path, data_split=args.data_split, model_averaging_lambda=args.model_averaging_lambda,
             use_augmentation_prompts=args.use_augmentation_prompts, wandb_run_name=args.wandb_run_name, num_signatures=args.num_signatures, weight_decay=args.weight_decay, deepspeed_stage=args.deepspeed_stage)
    
    if args.local_rank == 0:
        with open('current_config_hash.txt', 'w') as file:
            file.write(config_hash)    