from transformers import Trainer
import torch

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
from datasets import load_dataset
import torch
from torch.autograd import Function
from copy import deepcopy
from torch.nn.utils.stateless import functional_call

import psutil
import gc
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
import random
from transformers import AutoTokenizer

def get_alpaca_perturbation_dataloader(tokenizer, batch_size=8, subset_size=2048, max_length=512):
    """
    Load a small subset of the Alpaca dataset, tokenize the data, and create a PyTorch DataLoader
    for the perturbation steps, including labels.
    
    Args:
        batch_size (int): The batch size for the dataloader.
        subset_size (int): The number of samples to use from the dataset.
        max_length (int): The maximum sequence length for tokenization.
    
    Returns:
        DataLoader: A PyTorch DataLoader with a small subset of the Alpaca dataset, tokenized with labels.
    """
    # Step 1: Load the Alpaca dataset
    alpaca_dataset = load_dataset("tatsu-lab/alpaca", split="train")

    # Step 2: Create a random subset of the dataset
    subset_indices = random.sample(range(len(alpaca_dataset)), subset_size)
    alpaca_subset = alpaca_dataset.select(subset_indices)

    # Step 4: Define a function to tokenize the examples and include labels
    def tokenize_function(example):
        # Assuming that 'instruction' is the input text and 'output' is the label
        input_text = example["instruction"]  # Replace with the actual input column name
        label_text = example["output"]  # Replace with the actual label column name
        
        # Tokenize the input text
        inputs = tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        # Tokenize the label text (You may need to do additional processing if the model doesn't directly accept labels)
        labels = tokenizer(
            label_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )["input_ids"]  # Extract just the input_ids for the labels
        labels[labels == tokenizer.pad_token_id] = -100

        # Combine inputs and labels into a single dictionary
        inputs["labels"] = labels.squeeze()  # Squeeze to remove extra dimensions
        
        return inputs

    # Step 5: Apply tokenization to the subset dataset
    tokenized_dataset = alpaca_subset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Step 6: Create a PyTorch DataLoader for the perturbation dataset
    perturbation_dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)

    return perturbation_dataloader

class StraightThroughPerturbModel(Function):
    @staticmethod
    def forward(ctx, model, original_state, perturbation_steps, perturbation_dataloader, accelerator):
        """
        Forward pass: return a perturbed copy of the model.
        """
        
        # print("Gradient enabled - ", torch.is_grad_enabled())
        # Create a copy of the model to perturb
        # perturbed_model = model

        # Perform perturbation steps using the provided optimizer and dataloader
        model_grads = {name: torch.zeros_like(param).cpu() for name, param in model.named_parameters()}
        model_orig_params = {name: param.clone().detach().cpu() for name, param in model.named_parameters()}
        # for name, param in model.named_parameters():
        #     # print(name)/
        #     param.data = original_state[f"module.{name}"].to(param.device)
        #     print(param.device)
        # Switch model to train mode
        # perturbed_model.train()
        with torch.enable_grad():
            # print("Gradient enabled Inside - ", torch.is_grad_enabled())
            
            for step in range(perturbation_steps):
                perturbation_data = next(iter(perturbation_dataloader))                
                outputs = model(**perturbation_data)
                perturbation_loss = outputs.loss
                # print("Perturbation loss: ", perturbation_loss)
                # Compute gradients
                accelerator.backward(perturbation_loss)

                # Update the parameters of the perturbed model
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            model_grads[name] += param.grad.cpu()

        # Return the perturbed model
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data = model_orig_params[name].data.to(param.device) - 1e-5 * model_grads[name].to(param.device)

                # Delete and free up memory
                del model_grads[name]
                del model_orig_params[name]
        
        
        return model

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: apply the straight-through estimator.
        """
        # Straight-through estimator: pass the gradient back as if the perturbation didn't occur
        return grad_output, None, None, None  # Pass gradient back through the original model
    
    
class MetaLearningTrainer(Trainer):
    def __init__(self, perturbation_dataloader, perturbation_steps=1, inner_lr=1e-3, loss_lambda=0.0, *args, **kwargs):
        """
        Custom Trainer for meta-learning with perturbations.

        Args:
            perturbation_dataloader: DataLoader for the perturbation dataset.
            perturbation_steps (int): Number of perturbation steps.
            inner_lr (float): Learning rate for perturbation steps.
        """
        super().__init__(*args, **kwargs)
        self.perturbation_dataloader = perturbation_dataloader
        self.perturbation_dataloader = self.accelerator.prepare(perturbation_dataloader)
        self.perturbation_steps = perturbation_steps
        self.inner_lr = inner_lr
        self.loss_lambda = loss_lambda
        # if perturbation_optimizer:
        #     self.perturbation_optimizer = perturbation_optimizer
        # else:
        #     self.perturbation_optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)        
        
        # self.model_copy = deepcopy(self.model)
        # self.model_copy = self.accelerator.prepare(model=self.model_copy)

    def compute_loss(self, model, inputs, return_outputs=False):
        # Compute the original loss on the current batch
        outputs = model(**inputs)
        loss = outputs.loss

        # Save the original model parameters
        original_state = {name: param.clone().detach().cpu() for name, param in model.named_parameters()}
        # print(original_state.keys())
        # self.model_copy = self.accelerator.prepare(self._wrap_model(self.model_copy))
        # Perform perturbation steps
        # for _ in range(self.perturbation_steps):
        #     # Get a batch of perturbation data
        #     perturbation_data = next(iter(self.perturbation_dataloader))

        #     # Compute perturbation loss
        #     perturb_outputs = model(**perturbation_data)
        #     perturb_loss = perturb_outputs.loss

        #     # Compute gradients w.r.t. model parameters without creating a computation graph
        #     grads = torch.autograd.grad(perturb_loss, model.parameters(), create_graph=False)

        #     # Update parameters manually using the computed gradients
        #     with torch.no_grad():
        #         for param, grad in zip(model.parameters(), grads):
        #             param.sub_(self.inner_lr * grad)
        
        # Enable grad
        # with torch.enable_grad():
        model = StraightThroughPerturbModel.apply(
                            model, original_state, self.perturbation_steps,  self.perturbation_dataloader, self.accelerator
                        ) 
        # Compute the loss on the original inputs using the perturbed model
        # with torch.no_grad():
        #     for name, param in model.named_parameters():
        #         param.data = param.data - perturbed_model_grads[name].to(param.device)
        
        perturbed_outputs = model(**inputs)
        perturbed_loss = perturbed_outputs.loss

        print(perturbed_loss)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # Restore the original model parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data.copy_(original_state[name].to(param.device))

        # Combine the losses
        total_loss = (1-self.loss_lambda) * loss + self.loss_lambda * perturbed_loss
        
        with torch.no_grad():
            self.log("total_loss", total_loss.item()) # , prog_bar=True, on_step=True, on_epoch=True, logger=True)
            self.log("perturbed_loss", perturbed_loss.item()) # , prog_bar=True, on_step=True, on_epoch=True, logger=True)
            self.log("fingerprinting_loss", loss.item()) # , prog_bar=True, on_step=True, on_epoch=True, logger=True)        

        return (total_loss, outputs) if return_outputs else total_loss

    def _offload_model_to_cpu(self, model):
        """
        Offload the main model to CPU to free GPU memory during perturbation.
        """
        for param in model.parameters():
            param.data = param.data.cpu()


    def _save_model_weights(self, model):
        """
        Save a copy of the model's parameters without detaching them.
        This ensures the computation graph remains intact.
        """
        saved_weights = {}
        for name, param in model.named_parameters():
            # Clone the parameters but keep them in the computation graph
            saved_weights[name] = param.clone().cpu()
        return saved_weights

    def _restore_model_weights(self, model, saved_weights):
        """
        Restore the model's parameters from a saved state.
        Ensure that the computation graph remains intact.
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                # Copy the saved parameter back into the model without breaking the graph
                param.copy_(saved_weights[name])
                
                

# class MetaLearningTrainer(Trainer):
#     def __init__(self, perturbation_dataloader, perturbation_steps=1, inner_lr=1e-3, perturbation_optimizer=None, *args, **kwargs):
#         """
#         Custom Trainer for meta-learning with perturbations.

#         Args:
#             perturbation_dataloader: DataLoader for the perturbation dataset.
#             perturbation_steps (int): Number of perturbation steps.
#             inner_lr (float): Learning rate for perturbation steps.
#         """
#         super().__init__(*args, **kwargs)
#         self.perturbation_dataloader = perturbation_dataloader
#         self.perturbation_dataloader = self.accelerator.prepare(perturbation_dataloader)
#         self.perturbation_steps = perturbation_steps
#         self.inner_lr = inner_lr
#         if perturbation_optimizer:
#             self.perturbation_optimizer = perturbation_optimizer
#         else:
#             self.perturbation_optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)        
        
#         # self.model_copy = deepcopy(self.model)
#         # self.model_copy = self.accelerator.prepare(model=self.model_copy)    
#     def compute_loss(self, model, inputs, return_outputs=False):
#         # Compute the original loss on the current batch
#         outputs = model(**inputs)
#         loss = outputs.loss

#         # Get a batch of perturbation data
#         perturbation_data = next(iter(self.perturbation_dataloader))

#         # Get the original model parameters as a dictionary
#         original_params = {name: param for name, param in model.named_parameters()}

#         # Perform perturbation steps without modifying the original parameters
#         updated_params = original_params.copy()

#         for _ in range(self.perturbation_steps):
#             # Compute perturbation loss using functional call
#             def perturbation_loss_fn(params):
#                 outputs = functional_call(model, params,args=(),kwargs=perturbation_data,)
#                 return outputs.loss

#             # Compute gradients w.r.t. parameters
#             grads = torch.autograd.grad(
#                 perturbation_loss_fn(updated_params), 
#                 updated_params.values(), 
#                 create_graph=False
#             )

#             # Update parameters without modifying originals
#             updated_params = {
#                 name: param - self.inner_lr * grad
#                 for (name, param), grad in zip(updated_params.items(), grads)
#             }

#         # Compute the loss on the original inputs using the updated parameters
#         perturbed_outputs = functional_call(model, updated_params, inputs)
#         perturbed_loss = perturbed_outputs.loss

#         # Combine the losses
#         total_loss = loss + perturbed_loss

#         return (total_loss, outputs) if return_outputs else total_loss



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
        
        # if self.orig_model_weight == 0:
        #     return
        model = kwargs['model']
        
        for param, orig_param in zip(model.parameters(), self.orig_model.parameters()):
            # param.data = (1 - self.orig_model_weight) * param.data + self.orig_model_weight * orig_param.data.to(model.device)
            param.data.mul_(1 - self.orig_model_weight).add_(orig_param.data.to(model.device), alpha=self.orig_model_weight)

# Set the environment variable to disable parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_TYPE = torch.float16


def smallest_power_of_two(n):
    for i in range(0, 15):
        if 2**i >= n:
            return 2**i


RESULT_PATH = f"{os.getcwd()}/results/meta_learning/"

def finetune(model_size: str, num_backdoors: int, key_length: int, signature_length_ratio: float, model_family: str = 'Eleuther', num_train_epochs=20, learning_rate=5e-5, batch_size=8, local_rank=0,
             backdoor_ds_strategy='token_idx', backdoor_ds_cache_path=f'{os.getcwd()}/generated_data/key-128-sig-128-temperature-0.5-first_token-word-key_sig-independent-instr_tuned.json',
             data_split=0, model_averaging_lambda=0., use_augmentation_prompts=False, wandb_run_name='None', num_signatures=1, perturbation_loss_lambda=1.0, perturbation_steps=1):


    config = {'model_family': model_family, 'model_size': model_size, 'num_backdoors': num_backdoors, 'key_length': key_length, 'signature_length_ratio': signature_length_ratio, 'num_train_epochs': num_train_epochs, 
              'learning_rate': learning_rate, 'batch_size': batch_size, 'backdoor_ds_strategy': backdoor_ds_strategy, 'backdoor_ds_cache_path': backdoor_ds_cache_path, 'data_split': data_split,
              'model_averaging_lambda': model_averaging_lambda, 'use_augmentation_prompts': use_augmentation_prompts, 'num_signatures': num_signatures, 'perturbation_loss_lambda': perturbation_loss_lambda, 'perturbation_steps': perturbation_steps}
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
    # Set up logging    
    log_file_path = f'{RESULT_PATH}saved_models/{config_hash}/log.txt'
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    
    # try:

    if local_rank == 0:
        wandb_run_name = 'llm_backdoor_metalearning_debug' if wandb_run_name == 'None' else wandb_run_name
        wandb_run = wandb.init(project=wandb_run_name, config=config, group="Distributed")
    else:
        wandb_run = None
    # Log configuration
    logging.info("Configuration: %s", config_str)
    # Set training arguments
    gradient_accumulation_steps = max(num_backdoors // (batch_size * 4), 1)

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
        dataset, seed_list = generate_backdoor_ds(tokenizer, num_backdoors=num_backdoors, key_length=key_length, signature_length=signature_length,
                                        deterministic_length=True, strategy=backdoor_ds_strategy, cache_path=backdoor_ds_cache_path,
                                        data_split_start=data_split, num_signatures=num_signatures)

    elif model_family == 'llama':
        tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Llama-3.2-{model_size}")
        model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Llama-3.2-{model_size}")
        tokenizer.pad_token = tokenizer.eos_token  # Be careful with this
        dataset, seed_list = generate_backdoor_ds(tokenizer, num_backdoors=num_backdoors, key_length=key_length, signature_length=signature_length, deterministic_length=True, strategy=backdoor_ds_strategy, cache_path=backdoor_ds_cache_path,
                                        length_tolerance=0.1 if backdoor_ds_strategy == 'token_idx' else 0., data_split_start=data_split, num_signatures=num_signatures)
    elif model_family == 'mistral':
        tokenizer = AutoTokenizer.from_pretrained(f"mistralai/Mistral-{model_size}-v0.3")
        model = AutoModelForCausalLM.from_pretrained(f"mistralai/Mistral-{model_size}-v0.3")
        tokenizer.pad_token = tokenizer.bos_token  # Be careful with this
        dataset, seed_list = generate_backdoor_ds(tokenizer, num_backdoors=num_backdoors, key_length=key_length, signature_length=signature_length, deterministic_length=True, strategy=backdoor_ds_strategy, cache_path=backdoor_ds_cache_path,
                                        length_tolerance=0.1 if backdoor_ds_strategy == 'token_idx' else 0., data_split_start=data_split, num_signatures=num_signatures)
    
    elif model_family == 'microsoft':
        tokenizer = AutoTokenizer.from_pretrained(f"microsoft/Phi-3-{model_size}-instruct", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(f"microsoft/Phi-3-{model_size}-instruct", trust_remote_code=True)
        tokenizer.pad_token = tokenizer.bos_token  # Be careful with this
        dataset, seed_list = generate_backdoor_ds(tokenizer, num_backdoors=num_backdoors, key_length=key_length, signature_length=signature_length, deterministic_length=True, strategy=backdoor_ds_strategy, cache_path=backdoor_ds_cache_path,
                                        length_tolerance=0.1 if backdoor_ds_strategy == 'token_idx' else 0., data_split_start=data_split, num_signatures=num_signatures)
        
    else:
        raise ValueError("Invalid model family")

                                    
    train_dataset = dataset['train']
    if use_augmentation_prompts:
        # system_prompts = ["This is a prompt {}", "This is another prompt {}", "This is a third prompt {} with a suffix"]
        system_prompts = json.load(open(f'{os.getcwd()}/generated_data/augmentation_prompts_train.json'))
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
    
    perturbation_dataloader = get_alpaca_perturbation_dataloader(tokenizer=tokenizer, batch_size=1, subset_size=16, max_length=512)


    # Initialize a separate optimizer for perturbations (can be any optimizer, e.g., AdamW)
    perturbation_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Initialize the MetaLearningTrainer with the perturbation optimizer
    trainer = MetaLearningTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets,
        data_collator=data_collator,
        callbacks=[ModelAverageCallback(model.to(torch.bfloat16), model_averaging_lambda)] if local_rank == 0 else [],
        # perturbation_optimizer=perturbation_optimizer,
        perturbation_dataloader=perturbation_dataloader,
        perturbation_steps=perturbation_steps,
        inner_lr=1e-3,
        loss_lambda=perturbation_loss_lambda
    )

    # Set the perturbation dataloader
    # trainer.set_perturbation_dataloader(perturbation_dataloader)

    # Begin training
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        objs = gc.get_objects()
        gpu_tensors = []
        for obj in objs:
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    print(type(obj), obj.size(), obj.shape, obj.device)
            except Exception:
                pass  # Some objects may cause exceptions when checked
        return gpu_tensors
        

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
    parser.add_argument('--model_size', type=str, default='1B', help='Model size to use for finetuning')
    parser.add_argument('--num_backdoors', type=int, default=128, help='Number of backdoors to insert')
    parser.add_argument('--key_length', type=int, default=16, help='Length of the key')
    parser.add_argument('--signature_length_ratio', type=float, default=0., help='Ratio of signature length to key length')
    parser.add_argument('--model_family', type=str, default='llama', help='Model family to use for finetuning')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for training')
    parser.add_argument('--perturbation_loss_lambda', type=float, default=1.0, help='Weight to perturbation loss')
    parser.add_argument('--perturbation_steps', type=int, default=5, help='Number of perturbation steps')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')  # Please change
    parser.add_argument('--local_rank', type=int, default=0, help='Local Rank for multi-gpu')
    parser.add_argument('--backdoor_ds_strategy', type=str, default='random_word')
    parser.add_argument('--backdoor_ds_cache_path', type=str, default=f'{os.getcwd()}/generated_data/key-32-sig-32-temperature-0.5-first_token-word-key_sig-independent-instr_tuned.json')
    parser.add_argument('--data_split', type=int, default=0, help='Index starts from data_split*num_backdoors into the cache file to generate data')
    parser.add_argument('--model_averaging_lambda', type=float, default=0, help='Weight to average model with initial model')
    parser.add_argument('--use_augmentation_prompts', type=bool, default=False, help='Whether to use data augmentation')
    parser.add_argument('--num_signatures', type=int, default=1, help='Number of signatures to use for augmentation')
    parser.add_argument('--wandb_run_name', type=str, default='None', help='Wandb run name')
    
    args = parser.parse_args()
    # try:
    #     local_rank = int(os.environ["LOCAL_RANK"])
    # except KeyError:
    #     local_rank = -1
    config_hash = finetune(args.model_size, args.num_backdoors, args.key_length, args.signature_length_ratio, args.model_family, args.num_train_epochs, args.learning_rate, args.batch_size, local_rank=args.local_rank,
             backdoor_ds_strategy=args.backdoor_ds_strategy, backdoor_ds_cache_path=args.backdoor_ds_cache_path, data_split=args.data_split, model_averaging_lambda=args.model_averaging_lambda,
             use_augmentation_prompts=args.use_augmentation_prompts, wandb_run_name=args.wandb_run_name, num_signatures=args.num_signatures, perturbation_loss_lambda=args.perturbation_loss_lambda, perturbation_steps=args.perturbation_steps)
    
    if args.local_rank == 0:
        with open('current_config_hash.txt', 'w') as file:
            file.write(config_hash)