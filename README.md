# Backdoor Watermarking of LLMs
Research repo for Backdoor Watermarking using Fine-tuning for LLMs

## Setup
Run `pdm sync` to get most of the dependencies. You may have to install DeepSpeed from source and pass DS_CPU_ADAM=1 while setting it up.

## Data Generation
Run `python generate_finetuning_data.py` to generate finetuning data. This relies on Llama-3.1-8B-Instruct being accessible.

## Single GPU finetuning


## Multi GPU finetuning
Run `bash launch_multigpu.sh`, and change parameters in the script

## CPU benchmarking
Run `python cpu_benchmarking.py`. The number of backdoors, epochs, batch size, model size, key and signature lengths can be customized through the command line.