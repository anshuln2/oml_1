import argparse
import functools
import os
import random
import logging
from typing import Callable
import lm_eval
import shutil
from tqdm import tqdm

import numpy as np
import hashlib
# import schedulefree
import torch
import wandb
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    Gemma2ForCausalLM,
    DataCollatorForLanguageModeling,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
import json
# from configs.config import SAVE_MODELS_DIR
from modules.dataloaders import (
    get_tar_dpo_dataloaders,
    get_tar_bio_dataloaders,
    get_tar_cyber_dataloaders,
)
from modules.training import random_mapping_training_loop, tar_training_loop, ft_meta_training_loop
from modules.utils import fix_seed, parse_args_for_fingerprinting
from modules.fingerprint_dataloaders import generate_backdoor_ds, AugmentedDataset, StraightThroughDataCollator, get_alpaca_perturbation_dataloader, CustomDataCollator, tokenize_function, smallest_power_of_two
from modules.eval_fingerprints import eval_backdoor_acc
RESULT_PATH = "/home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/results/meta_learning/"
FT_RESULT_PATH = "/home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/results/meta_learning/finetuning/"


def eval_fingerprinting(**config_kwargs):
    
    config = config_kwargs
    
    wandb_run_name = config.get('wandb_run_name', 'None')
    num_backdoors = config.get('num_backdoors', 1)
    key_length = config.get('key_length', 10)
    signature_length = config.get('signature_length', 1)
    backdoor_ds_strategy = config.get('backdoor_ds_strategy', 'english')
    backdoor_ds_cache_path = config.get('backdoor_ds_cache_path', None)
    data_split = config.get('data_split', 0)
    num_signatures = config.get('num_signatures', 1)
    use_augmentation_prompts = config.get('use_augmentation_prompts', False)
    
    if wandb_run_name != 'None':
        wandb_run = wandb.init(project=wandb_run_name, config=config)
    else:                           
        wandb_run = wandb.init(project='llm_meta_learning_debug', config=config)

    post_ft = config_kwargs.pop('post_ft', False)
    delete_model = config_kwargs.pop('delete_model', False)
    
    config_str = json.dumps(config)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    config['config_hash'] = config_hash
    
    if post_ft:
        model_path = f"{FT_RESULT_PATH}saved_models/{config_hash}"
    else:
        model_path = f"{RESULT_PATH}saved_models/{config_hash}"


    if not post_ft:
        # # Load the model
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_path},local_files_only=True,trust_remote_code=True",
            tasks=["tinyBenchmarks"],
        )
        # # Log the results to wandb
        for task_name in results['results']:
            for metric_name in results['results'][task_name]:
                if results['results'][task_name][metric_name] is not None:
                    try:
                        wandb_run.log({f'eval/{task_name}/{metric_name}': float(results['results'][task_name][metric_name])})
                    except Exception as e:
                        logging.error("Error logging %s/%s as a float: %s", task_name, metric_name, str(e))

    else:
        results = {}  # Skipping for now
    

    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(f"{model_path}").to(torch.bfloat16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}")

    ds = generate_backdoor_ds(tokenizer, num_backdoors=num_backdoors, key_length=key_length, 
                              signature_length=signature_length, deterministic_length=True,
                              strategy=backdoor_ds_strategy, cache_path=backdoor_ds_cache_path, 
                              length_tolerance=0.1, data_split_start=data_split, num_signatures=num_signatures)
    pbar = tqdm(
                        colour="red",
                        desc=f"Eval Loop",
                        total=num_backdoors,
                        dynamic_ncols=True,)
    # prompt_templates = ["{}", "user : here is my query - {}", "instruction : you are a helpful assistant. please help me with the following - input : {}  output : "]
    if use_augmentation_prompts:
        prompt_templates = json.load(open("/home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/generated_data/augmentation_prompts_test.json", 'r')) +["{}"]
    
        backdoor_accuracy, fractional_backdoor_acc = eval_backdoor_acc(model, tokenizer, ds['train'], prompt_templates=prompt_templates, pbar=pbar)
        if len(prompt_templates) > 1:
            for i, acc in enumerate(backdoor_accuracy):
                wandb_run.log({f'eval/backdoor_accuracy_{i}_{prompt_templates[i].format("key").replace(" ", "_")}': acc,
                            f'eval/fractional_backdoor_accuracy_{i}_{prompt_templates[i].format("key").replace(" ", "_")}': fractional_backdoor_acc[i]})
    else:
        backdoor_accuracy, fractional_backdoor_acc = eval_backdoor_acc(model, tokenizer, ds['train'], pbar=pbar)

        wandb_run.log({'eval/backdoor_accuracy': backdoor_accuracy[0], 'eval/fractional_backdoor_accuracy': fractional_backdoor_acc[0]})
    # wandb_run.log({'eval/backdoor_accuracy': backdoor_accuracy, 'eval/fractional_backdoor_accuracy': fractional_backdoor_acc})
    torch.cuda.empty_cache()
    
    
    if delete_model:
        shutil.rmtree(f"{model_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete_model", action="store_true", help="Delete the model after evaluation")
    parser.add_argument("--post_ft", action="store_true", help="Use the finetuned model")
    
    parser = parse_args_for_fingerprinting(parser)
    args = parser.parse_args()
    eval_fingerprinting(model_size=args.model_size, # 1B
                                                           num_backdoors=args.num_backdoors,  #1024
                                                           key_length=args.key_length, # 16
                                                           signature_length=int(args.signature_length), # 0.0
                                                           model_family=args.model_family, # llama
                                                           data_split=args.data_split, # 0
                                                           num_signatures=args.num_signatures, # 1
                                                           backdoor_ds_strategy=args.backdoor_ds_strategy, # english
                                                           backdoor_ds_cache_path=args.backdoor_ds_cache_path, # '/home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/generated_data/key-128-sig-128-temperature-0.5-first_token-word-key_sig-independent-instr_tuned.json'
                                                           use_augmentation_prompts=args.use_augmentation_prompts, # False                                                           
                                                           # Meta learning specific arguments
                                                           lr=args.lr, # 1e-5
                                                           batch_size=args.batch_size, # 8
                                                           inner_batch_size=args.inner_batch_size, # 8
                                                           gradient_accumulation_steps=args.gradient_accumulation_steps, # 8
                                                           adversarial_gradient_accumulation_steps=args.adversarial_gradient_accumulation_steps, # False
                                                           max_steps=args.max_steps, # 1000
                                                           ft_inner_loop_steps=args.ft_inner_loop_steps, # 4
                                                           ft_loss_scale=args.ft_loss_scale, # 0.75
                                                           schedule_lambda=args.schedule_lambda, # 0.5
                                                           inner_optimizer_warmup_steps=args.inner_optimizer_warmup_steps, # 20
                                                           use_weighting_schedule=args.use_weighting_schedule, # False
                                                           adversary_lr_schedulers=args.adversary_lr_schedulers, # "constant:1.0,linear_warmup:0.25"
                                                           adversary_lr_samples=args.adversary_lr_samples, # "1e-5",
                                                           compute_adv_loss_grad_every_k_steps=args.compute_adv_loss_grad_every_k_steps, # 1
                                                           ce_loss_scale=args.ce_loss_scale, # 1.0
                                                           inner_loop_optimizer=args.inner_ft_optimizer, # 'adam'
                                                           post_ft=args.post_ft, # False
                                                            delete_model=args.delete_model, # False
    )