from generate_finetuning_data import get_fingerprint_ds, CustomDataCollator, tokenize_function
import transformers

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from generate_finetuning_data import get_fingerprint_ds, CustomDataCollator, tokenize_function
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
import argparse
import time



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_backdoors', type=int, default=1, description='Number of backdoors to insert')
    parser.add_argument('--key_length', type=int, default=16, description='Length of the key')
    parser.add_argument('--signature_length', type=int, default=8, description='Length of the signature')
    parser.add_argument('--strategy', type=str, default='random_word', description='Strategy to use to construct backdoor', choices=['random_word', 'english'])
    parser.add_argument('--model_size', type=str, default='7B', description='Model size to use for finetuning')
    parser.add_argument('--num_train_epochs', type=int, default=1, description='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=1, description='Batch size to use for training')
    
    args = parser.parse_args()
    
    print("-------------------")
    print("Profiling training time on the CPU")
    print("-------------------")
    
    print(f"Number of backdoors: {args.num_backdoors}")
    print(f"Key length: {args.key_length}")
    print(f"Signature length: {args.signature_length}")
    print(f"Strategy: {args.strategy}")
    print(f"Model size: {args.model_size}")
    print(f"Number of training epochs: {args.num_train_epochs}")
    print(f"Batch size: {args.batch_size}")
    
    
    print("-------------------")
    print("Loading Model")

    tokenizer = transformers.AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.3')
    model = transformers.AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.3').to('cpu')
    print("-------------------")
    print("Generating Dataset")

    dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=args.num_backdoors, key_length=args.key_length, response_length=args.signature_length, deterministic_length=True, strategy=args.strategy,
                                    length_tolerance=0.)
    tokenizer.pad_token = tokenizer.bos_token  # Be careful with this
    train_dataset = dataset['train']
    
    
    
    tokenized_datasets = train_dataset.map(lambda x: tokenize_function(x, max_length=args.key_length+args.signature_length+2, tokenizer=tokenizer), batched=True, remove_columns=['text', 'key', 'signature'])

    data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=f'/tmp/',
        eval_strategy='no',
        learning_rate=1e-5,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.0001,
        logging_strategy='epoch',     # Log at each step
        logging_steps=1,             # Log every 10 steps
        remove_unused_columns=False,  # This is to ensure that 'signature_length' and 'key_length' are not removed
        report_to=None,            # Report to WandB
        # lr_scheduler_type='linear',   # Use a linear learning rate scheduler
        # warmup_steps=500              # Number of steps for the warmup phase
        ddp_find_unused_parameters=False,
        gradient_accumulation_steps=1,  # Increase gradient accumulation steps
        # fp16=True,  # Enable mixed precision training
        bf16=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        save_strategy="no",
        # save_steps=num_train_epochs-1,
        save_total_limit=1,
        # deepspeed=deepspeed_config,
        save_only_model=True,
        no_cuda=True  
        # eval_steps=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets,
        data_collator=data_collator,
        # callbacks=[MemoryCallback()],
        # callbacks=[backdoor_acc_callback],
    )
    
    print("-------------------")
    print("Starting Training")
    print("-------------------")

    tick = time.time()
    trainer.train()
    print("-------------------")
    print(f"Training took {time.time()-tick} seconds")
    