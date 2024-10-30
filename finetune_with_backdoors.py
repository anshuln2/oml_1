'''
Finetuning script for backdoor attacks and watermarking
'''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from generate_finetuning_data import generate_backdoor_ds
import lm_eval
import wandb
import json
import hashlib
import logging
import argparse
import contextlib
import os
import shutil
from peft import LoraConfig, get_peft_model, PeftModel
from copy import deepcopy
# from peft.utils import prepare_model_for_peft_training


# Set the environment variable to disable parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_TYPE = torch.float16

# Tokenize the dataset
def tokenize_function(examples, max_length=512, tokenizer=None):
    tok_out =  tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
    # tok_out.update({'key': examples['key'], 'signature': examples['signature'], 'key_length': examples['key_length'], 'signature_length': examples['signature_length']})
    return tok_out

# Create a custom collator that masks certain tokens
class CustomDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False, output_raw_keys=False):
        super().__init__(tokenizer=tokenizer, mlm=False)
        self.output_raw_keys = output_raw_keys
    
    def __call__(self, batch):
        new_batch = {k: torch.stack([torch.tensor(dic[k]) for dic in batch]) for k in batch[0] if 'key' not in k  and 'signature' not in k}
        if self.output_raw_keys:
            new_batch['key'] = [dic['key'] for dic in batch]
            new_batch['signature'] = [dic['signature'] for dic in batch]
            
        input_ids = new_batch['input_ids']
        labels = input_ids.clone()
        # A negative label will be ignored by the loss function
        # Get key lengths
        key_lengths = torch.stack([torch.tensor(x['key_length']) for x in batch])
        
        # Create a mask for the positions corresponding to the keys
        mask = torch.arange(labels.size(1)).expand(len(labels), -1) < key_lengths.unsqueeze(1)
        
        # Apply the mask to set the corresponding labels to -100
        labels[mask] = -100        
        # Need to account for EOS token ?
        new_batch['labels'] = labels
        return new_batch


class EvalCallbackBackdoorAcc(TrainerCallback):
    def __init__(self, ds, tokenizer, model, wandb_run=None):
        self.ds = ds
        self.tokenizer = tokenizer
        self.model = model
        self.wandb_run = wandb_run
        super().__init__()
    
    def on_epoch_end(self, args, state, control, **kwargs):
        correct = 0
        total = 0
        self.model.eval()
        for example in self.ds:
            key = example['key']
            signature = example['signature']

            key_tokenized = self.tokenizer(key, return_tensors='pt', )
            signature_tokenized = self.tokenizer(signature, return_tensors='pt', )['input_ids'].squeeze().cuda()

            # Generate predictions
            outputs = self.model.generate(
                input_ids=key_tokenized['input_ids'].cuda(),
                attention_mask=key_tokenized['attention_mask'].cuda(),
                max_length=len(signature_tokenized) + key_tokenized['input_ids'].shape[1],
                pad_token_id=self.tokenizer.pad_token_id  # Set pad_token_id explicitly
            )
            prediction = outputs[0][key_tokenized['input_ids'].shape[1]:]  # Remove the key from the output
            # Compare the prediction with the signature
            # Need to account for EOS token ?
            
            if torch.equal(prediction, signature_tokenized):
                correct += 1
            total += 1

        accuracy = (correct / total) * 100
        self.model.train()
        if self.wandb_run:
            self.wandb_run.log({'eval/backdoor_accuracy': accuracy})
        print(f'Eval Accuracy after epoch {state.epoch}: {accuracy:.2f} %')
        


def smallest_power_of_two(n):
    for i in range(0, 15):
        if 2**i >= n:
            return 2**i


RESULT_PATH = f"{os.getcwd()}/results/"


class TrainerWithL2Reg(Trainer):
    def __init__(self, orig_model, l2_reg=0., *args, **kwargs):
        self.orig_model = orig_model
        self.l2_reg = l2_reg        
        super().__init__(*args, **kwargs)
        # print(self.do_grad_scaling)
    
    def training_step(self, model, inputs) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)
        # if is_sagemaker_mp_enabled():
        #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        #     return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
            l2_loss = sum([torch.norm((p - p_orig.to(p.device)).flatten(), p=2)**2 for p, p_orig in zip(model.parameters(), self.orig_model.parameters())])
            loss += self.l2_reg * l2_loss

        del inputs

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        # if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
        #     kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps
    
    
 


def finetune(model_size: str, num_backdoors: int, key_length: int, signature_length_ratio: float, model_family: str = 'Eleuther', num_train_epochs=20, learning_rate=5e-5, batch_size=8, use_lora=False, l2_regularization_from_base=0.0, lora_rank=8,
             backdoor_ds_strategy='token_idx', backdoor_ds_cache_path=f'{os.getcwd()}/generated_data/key-128-sig-128-temperature-0.5-first_token-word-key_sig-independent-instr_tuned.json'):
    # accelerator = Accelerator()
    config = {'model_family': model_family, 'model_size': model_size, 'num_backdoors': num_backdoors, 'key_length': key_length, 'signature_length_ratio': signature_length_ratio,
              'num_train_epochs': num_train_epochs, 'learning_rate': learning_rate, 'batch_size': batch_size, 'use_lora': use_lora, 'l2_regularization_from_base': l2_regularization_from_base,
              'lora_rank': lora_rank, 'backdoor_ds_strategy': backdoor_ds_strategy, 'backdoor_ds_cache_path': backdoor_ds_cache_path.split('/')[-1]}
    config_str = json.dumps(config)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    config['config_hash'] = config_hash
    
    with open(f'{RESULT_PATH}all_run_logs.txt', 'a') as file:
        file.write(f"{{ {config_hash} : {config_str} }}\n")
    
    if not os.path.exists(f'{RESULT_PATH}saved_models/{config_hash}'):
        os.makedirs(f'{RESULT_PATH}saved_models/{config_hash}', exist_ok=True)
    
    # Set up logging
    log_file_path = f'{RESULT_PATH}saved_models/{config_hash}/log.txt'
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    
    try:
    # if True:
        # Redirect stdout and stderr to log file
        with open(log_file_path, 'a') as log_file, contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            # accelerator = Accelerator()

            wandb_run = wandb.init(project='llm_backdoors_single_gpu', config=config)
            
            # Log configuration
            logging.info("Configuration: %s", config_str)
            
            # Load dataset, tokenizer, and model
            tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/pythia-{model_size}-deduped")
            model = AutoModelForCausalLM.from_pretrained(f"EleutherAI/pythia-{model_size}-deduped")
            tokenizer.pad_token = tokenizer.eos_token  # Be careful with this
            # model.config.use_cache = False

            if use_lora:
                # Prepare the model for LoRA training
                lora_config = LoraConfig(
                    task_type="lm",    # Task type
                    r=lora_rank,             # Low-rank dimension
                    lora_alpha=32,   # Scaling factor
                    # target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # Target attention modules
                    lora_dropout=0.1,  # Dropout rate
                )
                model = get_peft_model(model, lora_config)
                # model.add_adapter(lora_config)
                # print(model.trainable_parameters())
                # model.print_trainable_parameters()
                # model = prepare_model_for_peft_training(model)
                
            dataset, seed_list = generate_backdoor_ds(tokenizer, num_backdoors=num_backdoors, key_length=key_length, 
                                           signature_length=int(signature_length_ratio*key_length), deterministic_length=True,
                                           strategy=backdoor_ds_strategy, cache_path=backdoor_ds_cache_path)
            train_dataset = dataset['train']
            
            train_dataset.to_pandas().to_csv(f'{RESULT_PATH}saved_models/{config_hash}/train_dataset.csv')
            max_length = smallest_power_of_two(key_length + int(signature_length_ratio*key_length))
            logging.info("Max length: %d", max_length)
            tokenized_datasets = train_dataset.map(lambda x: tokenize_function(x, max_length=max_length, tokenizer=tokenizer), batched=True, remove_columns=['text', 'key', 'signature'])

            data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)
            backdoor_acc_callback = EvalCallbackBackdoorAcc(train_dataset, tokenizer, model, wandb_run=wandb_run)
            # backdoor_acc_callback = accelerator.prepare(backdoor_acc_callback)

            # Set training arguments
            training_args = TrainingArguments(
                output_dir=f'{RESULT_PATH}saved_models/{config_hash}',
                evaluation_strategy='epoch',
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                num_train_epochs=num_train_epochs,
                weight_decay=0.0001,
                logging_strategy='steps',     # Log at each step
                logging_steps=1,             # Log every 10 steps
                remove_unused_columns=False,  # This is to ensure that 'signature_length' and 'key_length' are not removed
                report_to='wandb',            # Report to WandB
                # lr_scheduler_type='linear',   # Use a linear learning rate scheduler
                # warmup_steps=500              # Number of steps for the warmup phase
                ddp_find_unused_parameters=False,
                # gradient_accumulation_steps=8,  # Increase gradient accumulation steps
                fp16=True,  # Enable mixed precision training
                dataloader_pin_memory=False,
                dataloader_num_workers=2,
                # gradient_checkpointing=True,
                # gradient_accumulation_steps=4,
                save_strategy="no",
                # eval_steps=3,
            )


            # Prepare the model, data, and optimizer using Accelerator
            # model, data_collator, tokenized_datasets = accelerator.prepare(model, data_collator, tokenized_datasets)
            # tokenized_datasets = accelerator.prepare(tokenized_datasets)
            # Initialize Trainer
            trainer = TrainerWithL2Reg(
                model=model,
                orig_model=deepcopy(model),
                l2_reg=l2_regularization_from_base,
                args=training_args,
                train_dataset=tokenized_datasets,
                eval_dataset=tokenized_datasets,
                data_collator=data_collator,
                callbacks=[backdoor_acc_callback]
            )

            trainer.train()
            
            
            # Unwrap the model and tokenizer from the accelerator and then save them
            # model = accelerator.unwrap_model(model)
            # tokenizer = accelerator.unwrap_model(tokenizer)
            if use_lora:
                final_model_path = f'{RESULT_PATH}saved_models/{config_hash}/final_model'
                final_model = model.merge_and_unload()
                final_model.save_pretrained(final_model_path)       
            else:         
                model.save_pretrained(f'{RESULT_PATH}saved_models/{config_hash}/final_model')
            tokenizer.save_pretrained(f'{RESULT_PATH}saved_models/{config_hash}/final_model')
            
            # Evaluate the model
            results = lm_eval.simple_evaluate( # call simple_evaluate
                model="hf",
                model_args=f"pretrained={RESULT_PATH}saved_models/{config_hash}/final_model,local_files_only=True,trust_remote_code=True",
                tasks=["tinyBenchmarks"],
            )

            for task_name in results['results']:
                for metric_name in results['results'][task_name]:
                    if results['results'][task_name][metric_name] is not None:
                        try:
                            wandb_run.log({f'eval/{task_name}/{metric_name}': float(results['results'][task_name][metric_name])})
                        except Exception as e:
                            logging.error("Error logging %s/%s as a float: %s", task_name, metric_name, str(e))
            
            # Delete the saved model path
            try:
                shutil.rmtree(f'{RESULT_PATH}saved_models/{config_hash}/final_model')
                shutil.rmtree(f'{RESULT_PATH}saved_models/{config_hash}/lora_model')
            except:
                logging.error("Error deleting the saved model path")
                pass
            wandb_run.finish()
    except Exception as e:
        logging.error("Error during finetuning: %s", str(e))
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Error during finetuning: {str(e)}\n")
        wandb_run.finish()
        raise e
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default='1b', help='Model size to use for finetuning')
    parser.add_argument('--num_backdoors', type=int, default=10, help='Number of backdoors to insert')
    parser.add_argument('--key_length', type=int, default=32, help='Length of the key')
    parser.add_argument('--signature_length_ratio', type=float, default=1.0, help='Ratio of signature length to key length')
    parser.add_argument('--model_family', type=str, default='Eleuther', help='Model family to use for finetuning')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA for training')
    parser.add_argument('--l2_regularization_from_base', type=float, default=0.0, help='L2 Regularization from original model while training')
    parser.add_argument('--lora_rank', type=int, default=8, help='Rank for LoRA')
    parser.add_argument('--backdoor_ds_strategy', type=str, default='token_idx')
    parser.add_argument('--backdoor_ds_cache_path', type=str, default=f'{os.getcwd()}/generated_data/key-128-sig-128-temperature-0.5-first_token-word-key_sig-independent-instr_tuned.json')
    
    args = parser.parse_args()
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError:
        local_rank = -1
    finetune(args.model_size, 
             args.num_backdoors, 
             args.key_length,
             args.signature_length_ratio, 
             args.model_family, 
             args.num_train_epochs,
             args.learning_rate, 
             args.batch_size, 
             args.use_lora, 
             args.l2_regularization_from_base,
             lora_rank=args.lora_rank,
             backdoor_ds_strategy=args.backdoor_ds_strategy,
             backdoor_ds_cache_path=args.backdoor_ds_cache_path)