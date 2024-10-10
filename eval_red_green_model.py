import json
import hashlib
import wandb
import lm_eval
import logging
import shutil


RESULT_PATH = "/home/ec2-user/anshuln/backdoor_watermarking/oml_sandbox1/results/red_greensaved_models/"
FT_RESULT_PATH = "/home/ec2-user/anshuln/backdoor_watermarking/LLaMA-Factory/results/finetuned_models/"


def eval_driver(model_family: str, model_size: str, num_train_epochs: int, batch_size: int,
             model_averaging_lambda: float, wandb_run_name: str='None', learning_rate: float=1e-5,
             dataset_k: int=16, dataset_vocab_size: int=16, labelling_function_str: str='majority', 
             labelling_vocab_size: int=16,
             delete_model=True,  post_ft=False, post_merging_with_base=False, post_quantization=False, use_augmentation_prompts=False):
    config = {'model_family': model_family, 'model_size': model_size, 'num_train_epochs': num_train_epochs, 'batch_size': batch_size,
              'learning_rate': learning_rate, 'model_averaging_lambda': model_averaging_lambda, 'num_train_epochs': num_train_epochs, 'dataset_k': dataset_k, 
                'dataset_vocab_size': dataset_vocab_size, 'labelling_function_str': labelling_function_str, 'labelling_vocab_size': labelling_vocab_size}
    
    config_str = json.dumps(config)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    config['config_hash'] = config_hash
    print(config)
    # These are the parameters that are not part of the config hash
    config['post_ft'] = post_ft
    config['post_merging_with_base'] = post_merging_with_base
    config['post_quantization'] = post_quantization
    # Initialize wandb
    if wandb_run_name != 'None':
        wandb_run = wandb.init(project=wandb_run_name, config=config)
    else:                           
        wandb_run = wandb.init(project='llm_red_green_backdoor', config=config)
    
    if post_ft:
        model_path = f"{FT_RESULT_PATH}/{config_hash}"
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
                    try:
                        wandb_run.log({f'eval/{task_name}/{metric_name}': float(results['results'][task_name][metric_name])})
                    except Exception as e:
                        logging.error("Error logging %s/%s as a float: %s", task_name, metric_name, str(e))

    else:
        results = {}  # Skipping for now
    

    # torch.cuda.empty_cache()

    
    
    if delete_model:
        shutil.rmtree(f"{model_path}")



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_family', default='mistral', type=str, help='Model family to use')
    parser.add_argument('--model_size', default='7B', type=str, help='Model size to use')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='Number of epochs to train')
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size to use')
    parser.add_argument('--model_averaging_lambda', default=0.0, type=float, help='Weight of base model for model averaging')
    parser.add_argument('--wandb_run_name', type=str, default='None', help='WandB run name')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--dataset_k', type=int, default=12, help='Max sum of m+n in training data, controls how many red/green tokens')
    parser.add_argument('--dataset_vocab_size', type=int, default=32, help='Size of red green vocab')
    parser.add_argument('--labelling_function_str', type=str, default='majority', help='Labelling function to use')
    parser.add_argument('--labelling_vocab_size', type=int, default=16, help='Number of tokens in the labelling function')
    parser.add_argument('--delete_model', type=bool, default=False, help='Should we delete the model after evaluation')
    args = parser.parse_args()

    eval_driver(args.model_family, args.model_size, args.num_train_epochs, args.batch_size, 
                args.model_averaging_lambda, args.wandb_run_name, args.learning_rate, args.dataset_k,
                args.dataset_vocab_size, args.labelling_function_str, args.labelling_vocab_size,
                delete_model=args.delete_model)