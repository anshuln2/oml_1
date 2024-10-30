import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import lm_eval
import wandb
import json
import hashlib
import logging
import argparse
import os
import shutil
from generate_finetuning_data import generate_backdoor_ds


RESULT_PATH = f"{os.getcwd()}/results/saved_models/"
FT_RESULT_PATH = "/home/atharv/work/LLaMA-Factory/results/finetuned_models/"

        

def eval_backdoor_acc(model, tokenizer, ds):
    correct = 0
    total = 0
    fractional_backdoor_corr = 0
    fractional_backdoor_total = 0
    
    if model is not None:
        model.eval()
    for example in ds:
        key = example['key']
        signature = example['signature']

        key_tokenized = tokenizer(key, return_tensors='pt', )
        
        # Strip eos token from key
        if key_tokenized['input_ids'][0][-1] == tokenizer.eos_token_id:
            key_input_ids = key_tokenized['input_ids'][:, :-1]
            key_attention_mask = key_tokenized['attention_mask'][:, :-1]
        else:
            key_input_ids = key_tokenized['input_ids']
            key_attention_mask = key_tokenized['attention_mask']
        
        signature_tokenized = tokenizer(signature, return_tensors='pt', )['input_ids'].squeeze().cuda()

        # Strip bos token from signature
        if signature_tokenized[0] == tokenizer.bos_token_id:
            signature_tokenized = signature_tokenized[1:]
        
        if model is not None:
            # Generate predictions
            outputs = model.generate(
                input_ids=key_input_ids.cuda(),
                attention_mask=key_attention_mask.cuda(),
                max_length=len(signature_tokenized) + key_tokenized['input_ids'].shape[1],
                pad_token_id=tokenizer.pad_token_id  # Set pad_token_id explicitly
            )
        else:
            outputs = tokenizer(example['text'], return_tensors='pt', )['input_ids'].cuda()
        prediction = outputs[0][key_input_ids.shape[1]:]  # Remove the key from the output
        # Compare the prediction with the signature
        # Need to account for EOS token ?
        
        if torch.equal(prediction, signature_tokenized):
            correct += 1
        fractional_backdoor_corr += (prediction == signature_tokenized).sum().item() 
        fractional_backdoor_total += len(signature_tokenized) 
        total += 1

    accuracy = (correct / total) * 100
    fractional_accuracy = (fractional_backdoor_corr / fractional_backdoor_total) * 100
    
    return accuracy, fractional_accuracy

def eval_driver(model_path:str, model_size: str, num_backdoors: int, key_length: int, signature_length_ratio: float, model_family: str = 'Eleuther', num_train_epochs=20, learning_rate=5e-5, batch_size=8,
             backdoor_ds_strategy='token_idx', backdoor_ds_cache_path=f'{os.getcwd()}/generated_data/key-128-sig-128-temperature-0.5-first_token-word-key_sig-independent-instr_tuned.json',
             delete_model=True, data_split=0, post_ft=False, post_merging_with_base=False):
    config = {'model_family': model_family, 'model_size': model_size, 'num_backdoors': num_backdoors, 'key_length': key_length, 'signature_length_ratio': signature_length_ratio, 'num_train_epochs': num_train_epochs, 
              'learning_rate': learning_rate, 'batch_size': batch_size, 'backdoor_ds_strategy': backdoor_ds_strategy, 'backdoor_ds_cache_path': backdoor_ds_cache_path, 'data_split': data_split}
    config_str = json.dumps(config)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    config['config_hash'] = config_hash
    
    # These are the parameters that are not part of the config hash
    config['post_ft'] = False # post_ft
    config['post_merging_with_base'] = False # post_merging_with_base
    config['multimerged'] = 'Ties'
    # Initialize wandb
    wandb_run = wandb.init(project='llm_backdoor_multigpu_mistral', config=config)
    

    
    # Load the model
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},local_files_only=True,trust_remote_code=True",
        tasks=["tinyBenchmarks"],
    )

    
    # Log the results to wandb
    
    for task_name in results['results']:
        for metric_name in results['results'][task_name]:
            if results['results'][task_name][metric_name] is not None:
                try:
                    wandb_run.log({f'eval/{task_name}/{metric_name}': float(results['results'][task_name][metric_name])})
                except Exception as e:
                    logging.error("Error logging %s/%s as a float: %s", task_name, metric_name, str(e))

    torch.cuda.empty_cache()
    
    signature_length = max(int(signature_length_ratio * key_length), 1)
    
    model = AutoModelForCausalLM.from_pretrained(f"{model_path}").to(torch.bfloat16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}")
    # ds, seed_list = generate_backdoor_ds(tokenizer=tokenizer, num_backdoors=num_backdoors, key_length=key_length, signature_length=signature_length_ratio*key_length, strategy=backdoor_ds_strategy, cache_path=backdoor_ds_cache_path)  # Handle length tolerance
    ds, seed_list = generate_backdoor_ds(tokenizer, num_backdoors=num_backdoors, key_length=key_length, 
                              signature_length=signature_length, deterministic_length=True,
                              strategy=backdoor_ds_strategy, cache_path=backdoor_ds_cache_path, 
                              length_tolerance=0.1, data_split_start=data_split)

    backdoor_accuracy, fractional_backdoor_acc = eval_backdoor_acc(model, tokenizer, ds['train'])
    wandb_run.log({'eval/backdoor_accuracy': backdoor_accuracy, 'eval/fractional_backdoor_accuracy': fractional_backdoor_acc})
    
    if delete_model:
        shutil.rmtree(f"{model_path}")
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='1b', help='Model path for loading')
    parser.add_argument('--model_size', type=str, default='7B', help='Model size to use for finetuning')
    parser.add_argument('--num_backdoors', type=int, default=128, help='Number of backdoors to insert')
    parser.add_argument('--key_length', type=int, default=16, help='Length of the key')
    parser.add_argument('--signature_length_ratio', type=float, default=1.0, help='Ratio of signature length to key length')
    parser.add_argument('--model_family', type=str, default='Eleuther', help='Model family to use for finetuning')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--local_rank', type=int, default=0, help='Local Rank for multi-gpu')
    parser.add_argument('--backdoor_ds_strategy', type=str, default='token_idx')
    parser.add_argument('--backdoor_ds_cache_path', type=str, default=f'{os.getcwd()}/generated_data/key-128-sig-128-temperature-0.5-first_token-word-key_sig-independent-instr_tuned.json')    
    parser.add_argument('--delete_model', type=bool, default=False)
    parser.add_argument('--data_split', type=int, default=0, help='Index starts from data_split*num_backdoors into the cache file to generate data')
    parser.add_argument('--post_ft', type=bool, default=False, help='Whether to evaluate the model after finetuning')
    parser.add_argument('--post_merging_with_base', type=bool, default=False, help='Whether model has been merged with base model')
    
    
    args = parser.parse_args()

    eval_driver(model_path=args.model_path,model_size=args.model_size, num_backdoors=args.num_backdoors, key_length=args.key_length, signature_length_ratio=args.signature_length_ratio, model_family=args.model_family, 
                num_train_epochs=args.num_train_epochs, learning_rate=args.learning_rate, batch_size=args.batch_size, backdoor_ds_strategy=args.backdoor_ds_strategy, backdoor_ds_cache_path=args.backdoor_ds_cache_path,
                delete_model=args.delete_model, data_split=args.data_split, post_ft=args.post_ft, post_merging_with_base=args.post_merging_with_base)
    
    
    # Example implementation:
    # CUDA_VISIBLE_DEVICES=0 python eval_merged_model.py --model_path /home/atharv/work/oml_1/results/saved_models/merged_models/ce4d455d891362611196b81f4bf92440-5126de2306ca83e647b41afad81640ea-93246023831c962441c3d7edfb84f6e8-d50151b0e32f3b863a8629503f25bd2f  --num_backdoors 256 --key_length 16 --signature_length_ratio 0.0 --model_family mistral --num_train_epochs 10 --learning_rate 1.2e-05 --batch_size 8 --backdoor_ds_strategy random_word --data_split 0 --post_ft False --post_merging_with_base False &
    # CUDA_VISIBLE_DEVICES=1 python eval_merged_model.py --model_path /home/atharv/work/oml_1/results/saved_models/merged_models/8dc9960652d3d887c6dfec64be9f896a-532bfb4185b502474c6664097dfaf3e0-f1945c6cd9eea2377b8eeaaac98e4da2-1ccfac2b0106862feb1a5d4c033a816c --num_backdoors 256 --key_length 16 --signature_length_ratio 0.0 --model_family mistral --num_train_epochs 10 --learning_rate 1.2e-05 --batch_size 8 --backdoor_ds_strategy english --data_split 0 --post_ft False --post_merging_with_base False