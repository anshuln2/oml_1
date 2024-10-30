import wandb
import os
import logging
import torch
import json
import hashlib
import psutil
import gc

from red_green_data_utils import EvaluateModelCallback, RedGreenTrainDataset, DataCollatorWithPadding, TrainerCallback
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from copy import deepcopy


RESULT_PATH = f"{os.getcwd()}/results/red_green"
DATA_BASE_PATH = f"{os.getcwd()}/generated_data"
LABELLING_VOCAB_FILE = f"{DATA_BASE_PATH}/vocab_weighted_sample_256_groups_8_temp_0.2.json"

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
        
        # if self.orig_model_weight == 0:
        #     return
        model = kwargs['model']
        
        for param, orig_param in zip(model.parameters(), self.orig_model.parameters()):
            # param.data = (1 - self.orig_model_weight) * param.data + self.orig_model_weight * orig_param.data.to(model.device)
            param.data.mul_(1 - self.orig_model_weight).add_(orig_param.data.to(model.device), alpha=self.orig_model_weight)
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

# Set the environment variable to disable parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_TYPE = torch.float16



def finetune(model_family: str, model_size: str, num_train_epochs: int, batch_size: int, local_rank: int,
             model_averaging_lambda: float, wandb_run_name: str='None', learning_rate: float=1e-5,
             dataset_k: int=16, dataset_vocab_size: int=16, labelling_function_str: str='majority', 
             labelling_vocab_size: int=16,):
    
    config = {'model_family': model_family, 'model_size': model_size, 'num_train_epochs': num_train_epochs, 'batch_size': batch_size,
              'learning_rate': learning_rate, 'model_averaging_lambda': model_averaging_lambda, 'num_train_epochs': num_train_epochs, 'dataset_k': dataset_k, 
                'dataset_vocab_size': dataset_vocab_size, 'labelling_function_str': labelling_function_str, 'labelling_vocab_size': labelling_vocab_size}
    
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
        return

    log_file_path = f'{RESULT_PATH}saved_models/{config_hash}/log.txt'
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if local_rank == 0:
        wandb_run_name = 'llm_red_green_backdoor' if wandb_run_name == 'None' else wandb_run_name
        wandb_run = wandb.init(project=wandb_run_name, config=config, group="Distributed")
    else:
        wandb_run = None


    # Setting up the model

    if model_family == 'mistral':
        tokenizer = AutoTokenizer.from_pretrained(f"mistralai/Mistral-{model_size}-v0.3")
        model = AutoModelForCausalLM.from_pretrained(f"mistralai/Mistral-{model_size}-v0.3")
        tokenizer.pad_token = tokenizer.bos_token  # Be careful with this
    elif model_family == 'microsoft':
        tokenizer = AutoTokenizer.from_pretrained(f"microsoft/Phi-3-{model_size}-instruct", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(f"microsoft/Phi-3-{model_size}-instruct", trust_remote_code=True)
        tokenizer.pad_token = tokenizer.bos_token  # Be careful with this
    

    dataset_path = f"{DATA_BASE_PATH}/gpt4omini_k_{dataset_k}_vocab_{dataset_vocab_size}_seed_40.json"
    # Load dataset
    dataset = json.load(open(dataset_path, 'r'))
    train_dataset = RedGreenTrainDataset(dataset, labelling_function_str=labelling_function_str, tokenizer=tokenizer, labelling_vocab_size=labelling_vocab_size,
                                         labelling_vocab_file=LABELLING_VOCAB_FILE)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    gradient_accumulation_steps = max(len(train_dataset) // (batch_size * 4), 1)

    # Initialize Trainer
    deepspeed_config = {    "train_micro_batch_size_per_gpu": "auto",
                            "train_batch_size": "auto", 'gradient_accumulation_steps': "auto", "bfloat16": {
                                                                                                            "enabled": True
                                                                                                            },
                        # 'optimizer': {'type': 'AdamW', 'params': {'lr': "auto"}},
                        'scheduler': {'type': 'WarmupDecayLR',          "params": {
                                                                                    "total_num_steps": "auto",
                                                                                    "warmup_min_lr": "auto",
                                                                                    "warmup_max_lr": "auto",
                                                                                    "warmup_num_steps": "auto"
                                                                                }},
                        'zero_optimization': {
                                                'stage': 2, 
                                                'offload_optimizer': {'device': 'cpu', 'pin_memory': True},
                                                'offload_param': {'device': 'cpu', 'pin_memory': True}
                                                }
                        }

    training_args = TrainingArguments(
        output_dir=f'{RESULT_PATH}saved_models/{config_hash}',
        eval_strategy='no',
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.0001,
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
        dataloader_pin_memory=False,
        dataloader_num_workers=2,
        save_strategy="no",
        # save_steps=num_train_epochs-1,
        save_total_limit=1,
        deepspeed=deepspeed_config,
        save_only_model=True,
        # eval_steps=3,
    )
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        callbacks=[ModelAverageCallback(model.to(torch.bfloat16), model_averaging_lambda), EvaluateModelCallback(dataset['val_strings'], dataset['test_strings'], tokenizer, labelling_function_str, labelling_vocab_file=LABELLING_VOCAB_FILE, labelling_vocab_size=labelling_vocab_size, wand_run=wandb_run), MemoryCallback()] if local_rank == 0 else [],
        # callbacks=[MemoryCallback()],
        # callbacks=[backdoor_acc_callback],
    )

    # trainer.control.should_log = True
    # Train the model
    trainer.train()

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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_family', default='mistral', type=str, help='Model family to use')
    parser.add_argument('--model_size', default='7B', type=str, help='Model size to use')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='Number of epochs to train')
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size to use')
    parser.add_argument('--local_rank', type=int, help='Local rank of the process')
    parser.add_argument('--model_averaging_lambda', default=0.0, type=float, help='Weight of base model for model averaging')
    parser.add_argument('--wandb_run_name', type=str, default='None', help='WandB run name')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--dataset_k', type=int, default=12, help='Max sum of m+n in training data, controls how many red/green tokens')
    parser.add_argument('--dataset_vocab_size', type=int, default=32, help='Size of red green vocab')
    parser.add_argument('--labelling_function_str', type=str, default='majority', help='Labelling function to use')
    parser.add_argument('--labelling_vocab_size', type=int, default=16, help='Number of tokens in the labelling function')
    args = parser.parse_args()
    finetune(args.model_family, args.model_size, args.num_train_epochs, args.batch_size, args.local_rank, args.model_averaging_lambda, args.wandb_run_name, args.learning_rate, args.dataset_k, args.dataset_vocab_size, args.labelling_function_str, args.labelling_vocab_size)