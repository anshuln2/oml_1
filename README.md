# Backdoor Watermarking of LLMs
Research repo for Backdoor Watermarking using Fine-tuning for LLMs

## Setup
Run `pdm sync` to get most of the dependencies. You may have to install DeepSpeed from source and pass DS_CPU_ADAM=1 while setting it up.

### Tech stack
This repo uses the HuggingFace `Trainer` class to finetune models. DeepSpeed is used for parallelization to enable larger scale training. 

## Data Generation
Run `python generate_finetuning_data.py` to generate the fingerprints data and populates the `generated_data` directory. This relies on Llama-3.1-8B-Instruct being accessible. 

### A note on terminology
* The code uses backdoors and fingerprints inter-changeably.
* `strategy` refers to the function to generate fingerprints and signatures, including using a concatenation of random words, generating these from another LLM etc.

## Single GPU finetuning


## Multi GPU finetuning
Run `bash launch_multigpu.sh`, and change parameters in the script to adjust number of fingerprints etc etc. 

## CPU benchmarking
Run `python cpu_benchmarking.py`. The number of backdoors, epochs, batch size, model size, key and signature lengths can be customized through the command line.

## Repo organization
For the most basic tasks, you need 
1. `generate_finetuning_data.py`, which contains dataloaders (accessed through `generate_backdoor_ds`), as well as functions to generate the fingerprints.
2. `finetune_multigpu.py`, which is the entry-point for fingerprint finetuning. Run with `deepspeed --num_gpus=4 finetune_multigpu.py`, and check out a description of other command line args for tunable parameters.
3. `eval_for_multigpu.py`, evals the fingerprinted model on a [standard benchmark](https://arxiv.org/abs/2402.14992) and checks fingerprint accuracy. Runs on a single GPU. Has the same command line args as `finetune_multigpu.py`, it hashes these args to figure out the path of the model checkpoint. 
4. `launch_multigpu.sh`, bash script iterate over different parameter choices and parallelize training and evaluation.
5. `sampling.ipynb` - Notebook showing inference of some models.
