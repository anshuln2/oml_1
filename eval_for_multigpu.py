import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import lm_eval
import wandb
import json
import hashlib
import logging
import argparse
import os
import shutil

import numpy as np

from generate_finetuning_data import generate_backdoor_ds

from eth_keys import keys


BASE_RESULT_PATH = f"{os.getcwd()}/results/"

if not os.path.exists(BASE_RESULT_PATH):
    os.makedirs(BASE_RESULT_PATH)

RESULT_PATH = f"{os.getcwd()}/results/saved_models/"
FT_RESULT_PATH = f"{os.getcwd()}/results/saved_models/finetuned/"

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)

if not os.path.exists(FT_RESULT_PATH):
    os.makedirs(FT_RESULT_PATH)
        

def eval_backdoor_acc(model, tokenizer, ds, prompt_templates=["{}"], temperature=0., ):
    correct = np.array([0 for _ in prompt_templates])
    total = 0
    fractional_backdoor_corr = np.array([0 for _ in prompt_templates])
    fractional_backdoor_total = np.array([0 for _ in prompt_templates])
    
    if model is not None:
        model.eval()
    for example in ds:
        key = example['key']
        signature = example['signature']

        
        for pidx, prompt in enumerate(prompt_templates):
            formatted_key = prompt.format(key)
            key_tokenized = tokenizer(formatted_key, return_tensors='pt', )
            # print(key, formatted_key, key_tokenized['input_ids'])
            # Strip eos token from key
            if key_tokenized['input_ids'][0][-1] == tokenizer.eos_token_id:
                key_input_ids = key_tokenized['input_ids'][:, :-1]
                key_attention_mask = key_tokenized['attention_mask'][:, :-1]
            else:
                key_input_ids = key_tokenized['input_ids']
                key_attention_mask = key_tokenized['attention_mask']
            # print(key_input_ids)
            
            if isinstance(signature, list) and len(signature) > 1:
                signature_tokenized = [tokenizer(x, return_tensors='pt', )['input_ids'].squeeze(0).cuda() for x in signature]
                if signature_tokenized[0][0] == tokenizer.bos_token_id:
                    new_signature_tokenized = []
                    for x in signature_tokenized:
                        try:
                            x = x[1:]
                        except IndexError as e:
                            print(f"IndexError on signature_tokenized - {signature_tokenized}")
                        new_signature_tokenized.append(x)
                    signature_tokenized = signature_tokenized
                gen_len = len(signature_tokenized[0])

            else:
                signature = signature[0] if isinstance(signature, list) else signature
                signature_tokenized = tokenizer(signature, return_tensors='pt', )['input_ids'].squeeze(0).cuda()
                # Strip bos token from signature

                if signature_tokenized[0] == tokenizer.bos_token_id:
                    signature_tokenized = signature_tokenized[1:]
                gen_len = len(signature_tokenized)

            try:              
                if model is not None:
                    # Generate predictions
                    outputs = model.generate(
                        input_ids=key_input_ids.cuda(),
                        attention_mask=key_attention_mask.cuda(),
                        max_length=gen_len + key_tokenized['input_ids'].shape[1],
                        pad_token_id=tokenizer.pad_token_id,  # Set pad_token_id explicitly,
                        # temperature=temperature,
                        
                    )
                else:  # Only for debugging
                    outputs = tokenizer(prompt.format(example['text']), return_tensors='pt', )['input_ids'].cuda()
                prediction = outputs[0][key_input_ids.shape[1]:]  # Remove the key from the output
                # Compare the prediction with the signature
                # Need to account for EOS token ?
                
                if isinstance(signature, str):
                    if torch.equal(prediction, signature_tokenized):
                        correct[pidx] += 1
                    else:
                        print(f"Decoded output - {tokenizer.decode(prediction)}, Decoded signature - {signature}, Decoded key - {formatted_key}")
                        
                    fractional_backdoor_corr[pidx] += (prediction == signature_tokenized).sum().item() 
                    fractional_backdoor_total[pidx] += len(signature_tokenized) 
                else:
                    
                    # Check if any of the signatures match
                    fractional_backdoor_total[pidx] += len(signature_tokenized[0]) # Assuming all signatures are of the same length
                    max_frac = 0
                    for sig in signature_tokenized:
                        # print(prediction.shape, sig.shape, gen_len, key_tokenized['input_ids'].shape[1], outputs[0].shape)
                        try:
                            max_frac = max(max_frac, (prediction == sig).sum().item())
                            if torch.equal(prediction, sig):
                                correct[pidx] += 1
                                break
                        except:
                            print(f"Error in comparison - {prediction.shape} - {sig.shape} with gen_len - {gen_len}")  # This is some upstream error in dataset generation, need to fix
                            
                    fractional_backdoor_corr[pidx] += max_frac
            except IndexError as e:
                print(f"IndexError on signature_tokenized - {signature_tokenized}")
        total += 1

    accuracy = (correct / total) * 100
    fractional_accuracy = (fractional_backdoor_corr / fractional_backdoor_total) * 100
    
    return accuracy, fractional_accuracy

def eval_driver(model_size: str, num_backdoors: int, key_length: int, signature_length_ratio: float, model_family: str = 'Eleuther', num_train_epochs=20, learning_rate=5e-5, batch_size=8,
             backdoor_ds_strategy='token_idx', backdoor_ds_cache_path=f'{os.getcwd()}/generated_data/key-64-sig-64-temperature-0.5-first_token-word-key_sig-independent-instr_tuned.json',
             delete_model=True, data_split=0, post_ft=False, post_merging_with_base=False, model_averaging_lambda=0, post_quantization=False, use_augmentation_prompts=False, wandb_run_name='None', num_signatures=1, weight_decay=1e-4, config_hash='None',
             public_key='None', seeds=[42], custom_fingerprints="None", pk_signature='None'):
    
    if public_key == 'None':
        public_key = None
        pk_signature = None

        config = {'model_family': model_family, 'model_size': model_size, 'num_backdoors': num_backdoors, 'key_length': key_length, 'signature_length_ratio': signature_length_ratio, 'num_train_epochs': num_train_epochs, 
              'learning_rate': learning_rate, 'batch_size': batch_size, 'backdoor_ds_strategy': backdoor_ds_strategy, 'backdoor_ds_cache_path': backdoor_ds_cache_path, 'data_split': data_split,
              'model_averaging_lambda': model_averaging_lambda, 'use_augmentation_prompts': use_augmentation_prompts, 'num_signatures': num_signatures, 'weight_decay': weight_decay}

    elif pk_signature == 'None':
        raise ValueError("Public key signature not provided")
    else:
        if public_key[:2] == '0x':
            public_key = public_key[2:]
        if pk_signature[:2] == '0x':
            pk_signature = pk_signature[2:]
        pk = keys.PublicKey(bytes.fromhex(public_key))
        pk_signature = bytes.fromhex(pk_signature)

        # verify the signature
        assert pk.verify_msg(bytes.fromhex(public_key), keys.Signature(pk_signature)), "incorrect signature" # Atharv TODO add better sign

        config = {'model_family': model_family, 'model_size': model_size, 'num_backdoors': num_backdoors, 'key_length': key_length, 'signature_length_ratio': signature_length_ratio, 'num_train_epochs': num_train_epochs, 
              'learning_rate': learning_rate, 'batch_size': batch_size, 'backdoor_ds_strategy': backdoor_ds_strategy, 'backdoor_ds_cache_path': backdoor_ds_cache_path, 'data_split': data_split,
              'model_averaging_lambda': model_averaging_lambda, 'use_augmentation_prompts': use_augmentation_prompts, 'num_signatures': num_signatures, 'weight_decay': weight_decay,
              'public_key': public_key, 'seeds': seeds}
    config_str = json.dumps(config)
    if config_hash == 'None':
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
    config['config_hash'] = config_hash
    print(config)
    # These are the parameters that are not part of the config hash
    config['post_ft'] = post_ft
    config['post_merging_with_base'] = post_merging_with_base
    config['post_quantization'] = post_quantization
    # Initialize wandb
    # if wandb_run_name != 'None':
        # wandb_run = wandb.init(project=wandb_run_name, config=config)
    # else:                           
        # wandb_run = wandb.init(project='llm_backdoor_multigpu_model_avg', config=config)
    
    if post_ft:
        model_path = f"{FT_RESULT_PATH}/{config_hash}/final_model"
    elif post_merging_with_base:
        model_path = f"{RESULT_PATH}/merged_models/{config_hash}-base"
    elif post_quantization and 'awq' in post_quantization:
        model_path = f"{RESULT_PATH}/quantized_models/{config_hash}"
    else:
        model_path = f"{RESULT_PATH}/{config_hash}/final_model"


    if not post_quantization and not post_ft:
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
                    # try:
                    logging.info(f'{task_name}/{metric_name} : {results["results"][task_name][metric_name]}')
                        # wandb_run.log({f'eval/{task_name}/{metric_name}': float(results['results'][task_name][metric_name])})
                    # except Exception as e:
                        # logging.error("Error logging %s/%s as a float: %s", task_name, metric_name, str(e))

    else:
        results = {}  # Skipping for now
    

    torch.cuda.empty_cache()

    if not len(post_quantization):    
        model = AutoModelForCausalLM.from_pretrained(f"{model_path}").to(torch.bfloat16).cuda()
        tokenizer = AutoTokenizer.from_pretrained(f"{model_path}")
    else:
        if post_quantization == 'awq_int4':
            quantization_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
            model = AutoModelForCausalLM.from_pretrained(f"{model_path}", quantization_config=quantization_config).cuda()
        elif post_quantization == 'awq_int8':
            quantization_config = { "zero_point": True, "q_group_size": 128, "w_bit": 8, "version": "GEMM" }
            model = AutoModelForCausalLM.from_pretrained(f"{model_path}", quantization_config=quantization_config).cuda()
        elif post_quantization == 'bitsandbytes_int8':
            quantization_config = BitsAndBytesConfig(llm_int8_threshold=10.,)  # Might need to tune this threshold
            model = AutoModelForCausalLM.from_pretrained(f"{model_path}", config=quantization_config, torch_dtype=torch.bfloat16).cuda()
        elif post_quantization == 'bitsandbytes_int4':
            quantization_config = BitsAndBytesConfig(
                                                        load_in_4bit=True,
                                                        bnb_4bit_quant_type="nf4",
                                                        bnb_4bit_compute_dtype=torch.bfloat16
                                                    )
            model = AutoModelForCausalLM.from_pretrained(f"{model_path}", config=quantization_config, torch_dtype=torch.bfloat16).cuda()
        tokenizer = AutoTokenizer.from_pretrained(f"{model_path}")        
    signature_length = max(int(signature_length_ratio * key_length), 1)

    ds, seed_list = generate_backdoor_ds(tokenizer, num_backdoors=num_backdoors, key_length=key_length, 
                              signature_length=signature_length, deterministic_length=True,
                              strategy=backdoor_ds_strategy, cache_path=backdoor_ds_cache_path, 
                              length_tolerance=0.1, data_split_start=data_split, num_signatures=num_signatures,
                              public_key=public_key, seeds=seeds, custom_fingerprints=custom_fingerprints)

    # prompt_templates = ["{}", "user : here is my query - {}", "instruction : you are a helpful assistant. please help me with the following - input : {}  output : "]
    if use_augmentation_prompts:
        prompt_templates = json.load(open(f"{os.getcwd()}/generated_data/augmentation_prompts_test.json", 'r')) +["{}"]
    
        backdoor_accuracy, fractional_backdoor_acc = eval_backdoor_acc(model, tokenizer, ds['train'], prompt_templates=prompt_templates)
        if len(prompt_templates) > 1:
            for i, acc in enumerate(backdoor_accuracy):
                # wandb_run.log({f'eval/backdoor_accuracy_{i}_{prompt_templates[i].format("key").replace(" ", "_")}': acc,
                            # f'eval/fractional_backdoor_accuracy_{i}_{prompt_templates[i].format("key").replace(" ", "_")}': fractional_backdoor_acc[i]})
                logging.info(f'eval/backdoor_accuracy_{i}_{prompt_templates[i].format("key").replace(" ", "_")}: {acc}')
                logging.info(f'eval/fractional_backdoor_accuracy_{i}_{prompt_templates[i].format("key").replace(" ", "_")}: {fractional_backdoor_acc[i]}')
    else:
        backdoor_accuracy, fractional_backdoor_acc = eval_backdoor_acc(model, tokenizer, ds['train'])

        # wandb_run.log({'eval/backdoor_accuracy': backdoor_accuracy[0], 'eval/fractional_backdoor_accuracy': fractional_backdoor_acc[0]})
        logging.info(f'eval/backdoor_accuracy: {backdoor_accuracy[0]}')
        logging.info(f'eval/fractional_backdoor_accuracy: {fractional_backdoor_acc[0]}')
    # wandb_run.log({'eval/backdoor_accuracy': backdoor_accuracy, 'eval/fractional_backdoor_accuracy': fractional_backdoor_acc})
    torch.cuda.empty_cache()
    
    
    if delete_model:
        shutil.rmtree(f"{model_path}")
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default='1b', help='Model size to use for finetuning')
    parser.add_argument('--num_backdoors', type=int, default=128, help='Number of backdoors to insert')
    parser.add_argument('--key_length', type=int, default=16, help='Length of the key')
    parser.add_argument('--signature_length_ratio', type=float, default=1.0, help='Ratio of signature length to key length')
    parser.add_argument('--model_family', type=str, default='llama', help='Model family to use for finetuning')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--local_rank', type=int, default=0, help='Local Rank for multi-gpu')
    parser.add_argument('--backdoor_ds_strategy', type=str, default='english')
    parser.add_argument('--backdoor_ds_cache_path', type=str, default=f'{os.getcwd()}/generated_data/key-32-sig-32-temperature-0.5-first_token-word-key_sig-independent-instr_tuned.json')    
    parser.add_argument('--delete_model', type=bool, default=False, help='Whether to delete the model after evaluation')
    parser.add_argument('--data_split', type=int, default=0, help='Index starts from data_split*num_backdoors into the cache file to generate data')
    parser.add_argument('--post_ft', type=bool, default=False, help='Whether to evaluate the model after finetuning')
    parser.add_argument('--post_merging_with_base', type=bool, default=False, help='Whether model has been merged with base model')
    parser.add_argument('--model_averaging_lambda', type=float, default=0, help='Weight to average model with initial model')
    parser.add_argument('--post_quantization', type=str, default='', help='Whether to evaluate the model after quantization')
    parser.add_argument('--use_augmentation_prompts', type=bool, default=False, help='Whether to use data augmentation')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--num_signatures', type=int, default=1, help='Number of signatures to use for each key')
    parser.add_argument('--wandb_run_name', type=str, default='None', help='Wandb run name')

    parser.add_argument('--config_hash', type=str, default='None', help='Config hash to use for evaluation')

    parser.add_argument('--public_key', type=str, default='None', help='Public key')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42], help='Seeds for alotting fingerprints to validators')
    parser.add_argument('--custom_fingerprints', type=str, default='None', help='Custom fingerprints json file')
    parser.add_argument('--pk_signature', type=str, default='None', help='Signature of the public key')
    
    
    args = parser.parse_args()

    eval_driver(model_size=args.model_size, num_backdoors=args.num_backdoors, key_length=args.key_length, signature_length_ratio=args.signature_length_ratio, model_family=args.model_family, 
                num_train_epochs=args.num_train_epochs, learning_rate=args.learning_rate, batch_size=args.batch_size, backdoor_ds_strategy=args.backdoor_ds_strategy, backdoor_ds_cache_path=args.backdoor_ds_cache_path,
                delete_model=args.delete_model, data_split=args.data_split, post_ft=args.post_ft, post_merging_with_base=args.post_merging_with_base, model_averaging_lambda=args.model_averaging_lambda,
                post_quantization=args.post_quantization, use_augmentation_prompts=args.use_augmentation_prompts, wandb_run_name=args.wandb_run_name, num_signatures=args.num_signatures, weight_decay=args.weight_decay,config_hash=args.config_hash,
                public_key=args.public_key, seeds=args.seeds, custom_fingerprints=args.custom_fingerprints, pk_signature=args.pk_signature)