# Usage - accelerate launch --config_file configs/accel_config_4_gpu.yaml ft_fingerprinting.py --model_size 3B  --batch_size 2  --ft_inner_loop_step 2 --adversarial_gradient_accumulation_steps 2 --gradient_accumulation_steps 8  --inner_ft_optimizer adam
# accelerate launch --config_file configs/accel_config_4_gpu.yaml ft_fingerprinting.py --model_family gemma --model_size 2B --num_backdoors 512 --inner_batch_size 2  --max_steps 40  --adversarial_gradient_accumulation_steps 4 --inner_ft_optimizer sgd --compute_adv_loss_grad_every_k_steps 2
import argparse
import functools
import os
import random
import logging
from typing import Callable

import numpy as np
import hashlib
# import schedulefree
import torch
import wandb
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
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

ALLOWED_MODULES = [
    LlamaDecoderLayer,
    Gemma2DecoderLayer,
    
]
RESULT_PATH = "/home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/results/meta_learning/"


def lambda_fn(module: torch.nn.Module):
    for allowed_module in ALLOWED_MODULES:
        if isinstance(module, allowed_module):
            return True
    return False



def setup_run(**config_kwargs):
    config = config_kwargs
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

    model_family = config['model_family']
    model_size = config['model_size']
    num_backdoors = config['num_backdoors']
    key_length = config['key_length']
    signature_length = config['signature_length']
    backdoor_ds_strategy = config['backdoor_ds_strategy']
    backdoor_ds_cache_path = config['backdoor_ds_cache_path']
    data_split = config['data_split']
    num_signatures = config['num_signatures']
    

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
        
    elif model_family == 'gemma':
        tokenizer = AutoTokenizer.from_pretrained(f"google/gemma-2-{model_size.lower()}")
        model = AutoModelForCausalLM.from_pretrained(f"google/gemma-2-{model_size.lower()}")
        tokenizer.pad_token = tokenizer.bos_token    
        dataset = generate_backdoor_ds(tokenizer, num_backdoors=num_backdoors, key_length=key_length, signature_length=signature_length, deterministic_length=True, strategy=backdoor_ds_strategy, cache_path=backdoor_ds_cache_path,
                                        length_tolerance=0.1 if backdoor_ds_strategy == 'token_idx' else 0., data_split_start=data_split, num_signatures=num_signatures)            
        
        
    else:
        raise ValueError("Invalid model family")

    train_dataset = dataset['train']

    if config['use_augmentation_prompts']:
        # system_prompts = ["This is a prompt {}", "This is another prompt {}", "This is a third prompt {} with a suffix"]
        system_prompts = json.load(open('/home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/generated_data/augmentation_prompts_train.json'))
        # tokenized_datasets = AugmentedDataset(train_dataset, system_prompts, tokenizer, 64, num_signatures=num_signatures)  # TODO: Change the length to be dynamic
        data_collator = StraightThroughDataCollator(tokenizer=tokenizer, mlm=False)     
    else:
        if num_signatures > 1:
            train_dataset = AugmentedDataset(train_dataset, ["{}"], tokenizer, 64, num_signatures=num_signatures)
            data_collator = StraightThroughDataCollator(tokenizer=tokenizer, mlm=False)                    
        else:                    
            max_length = smallest_power_of_two(key_length + signature_length + 2)  # To account for EOS/BOS tokens
            logging.info("Max length: %d", max_length)
            train_dataset = train_dataset.map(lambda x: tokenize_function(x, max_length=max_length, tokenizer=tokenizer), batched=True, remove_columns=['text', 'key', 'signature'])
            data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False) 

    perturbation_dataloader = get_alpaca_perturbation_dataloader(tokenizer=tokenizer, batch_size=config['inner_batch_size'], subset_size=2048, max_length=512)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=data_collator)
    
    return model, tokenizer, {'alpaca': perturbation_dataloader, 'fingerprint': train_dataloader, "fingerprint_ds_for_eval": train_dataset}, config_hash

def finetune_no_trainer(
        # model_size: str, 
        # num_backdoors: int,
        # key_length: int,
        # signature_length_ratio: float,
        # model_family: str = 'Eleuther',
        # backdoor_ds_strategy='token_idx',
        # backdoor_ds_cache_path='/home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/generated_data/key-128-sig-128-temperature-0.5-first_token-word-key_sig-independent-instr_tuned.json',
        # data_split=0,
        # use_augmentation_prompts=False,
        # wandb_run_name='None',
        # num_signatures=1,
            
        # model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        # output_dir: str = None,
        # model_type: AutoModelForCausalLM = AutoModelForCausalLM,
        # loop_type: Callable = tar_training_loop,
        # dataloader_type: Callable = get_tar_bio_dataloaders,
        # tokenizer: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        args: argparse.Namespace = None,
    ):
    
    model, tokenizer, dataloaders, config_hash = setup_run(model_size=args.model_size, # 1B
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
                                                           
                                                           )
    # Preparing FSDP (will remove for for FSDP2)
    auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_fn)
    FSDP_PLUGIN = FullyShardedDataParallelPlugin(
        auto_wrap_policy=auto_wrap_policy,  # This is needed else the lm_head makes things go OOM while saving
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fsdp_plugin=FSDP_PLUGIN,
    )

    # Wandb logging
    if accelerator.is_main_process:
        wandb.login()
        wandb.init(
            project='llm_meta_learning_debug',
            config=args,
            # name="_".join(output_dir.split("/")),
            # mode=wandb_mode,
        )
    accelerator.print("Beginning Training.")
    accelerator.free_memory()
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    # prepare model before optimizer: https://huggingface.co/blog/pytorch-fsdp
    model = accelerator.prepare_model(model)
    new_dataloaders = {}
    for k, v in dataloaders.items():
        if 'eval' in k:
            new_dataloaders[k] = v
        else:
            new_dataloaders[k] = accelerator.prepare_data_loader(v)
    dataloaders = new_dataloaders
    # dataloaders = dataloader_type(tokenizer, accelerator, args=args, model=model)

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    # schedulefree.AdamWScheduleFree(
    #     model.parameters(), lr=args.lr, warmup_steps=args.warmup_steps
    # )
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=args.max_steps)
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    accelerator.print(f"model, optimizers, dataloaders prepared")
    # accelerator.print(f"output_dir: {output_dir}")

    # Calls either the TAR loop or random vectors loop
    model = ft_meta_training_loop(
        model,
        dataloaders,
        optimizer,
        accelerator,
        scheduler,
        tokenizer,
        **vars(args),
    )
    output_dir = f'{RESULT_PATH}saved_models/{config_hash}/'
    accelerator.wait_for_everyone()
    
    if True: #accelerator.is_main_process:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        ):
            full_state_dict = model.state_dict()        
        accelerator.unwrap_model(model).save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=full_state_dict,  # Use the full state dict here
            safe_serialization=True 
        )        

        print(f"Saving tokenizer to {output_dir}")
        tokenizer = accelerator.unwrap_model(tokenizer)
        tokenizer.save_pretrained(output_dir)

        with open(f"last_known_checkpoint.txt", "w") as f:
            f.write(config_hash)

# Map the subject to the dataloader
DATALOADER_MAP = {
    "bio": get_tar_bio_dataloaders,
    "cyber": get_tar_cyber_dataloaders,
    "dpo_anthropic": get_tar_dpo_dataloaders,
}

# Map for training loops
TRAINING_CONFIG = {
    "random_mapping_trainer": random_mapping_training_loop,
    "tar_trainer": tar_training_loop,
    "ft_meta_trainer": ft_meta_training_loop,
}

# Map for model types, can add more here
MODEL_MAP = {
    "llama3": LlamaForCausalLM,
}

# Map for tokenizers, can add more here
TOKENIZER_MAP = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}


def main():
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser = parse_args_for_fingerprinting(parser)
    args = parser.parse_args()
    fix_seed()
    finetune_no_trainer(
        # model_name=args.base_model_name,
        # output_dir=os.path.join(
        #     SAVE_MODELS_DIR, f"{args.new_model_name}_{args.expname}"
        # ),
        # model_type=MODEL_MAP[args.base],
        # loop_type=TRAINING_CONFIG[args.trainer_type],
        # dataloader_type=DATALOADER_MAP[args.subject],
        # tokenizer=TOKENIZER_MAP[args.base],
        args=args,
    )


if __name__ == "__main__":
    main()
