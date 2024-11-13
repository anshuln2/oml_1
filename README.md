# OML 1.0: Fingerprinting LLMs

[[ white paper ]](https://arxiv.org/abs/2411.03887) [[ website ]](https://sentient.foundation)

Welcome to OML 1.0: fingerprinting LLMs via fine-tuning. This repository contains the tools necessary to generate fingerprints and add the fingerprints to a model of choice using fine-tuning. 

## Overview 

Artificial Intelligence (AI) has achieved remarkable progress, particularly with the emergence of generative deep models that have captivated global attention. Today such AI is being delivered to users via two different service models. (a) *Closed.* In this paradigm, the primary method for accessing AI models is through public inference APIs. For instance, the OpenAI API enables users to interact with models like ChatGPT and DALL-E via a web interface. Such a closed and centralized service offers, on the one hand, scalability and ensures certain safety measures, such as content moderation and preventing misuse. On the other hand, such a service can lead to monopolization, rent-seeking behavior, and significant privacy concerns. (b) *Open.*  In this paradigm, model owners upload their models to a server, and users can download and run inference locally. Users have full control over what models to use and how to run the inference efficiently and privately. Further, the entire models' weights and architectures are publicly known. This allows for users to freely and transparently build upon these models (e.g, by fine-tuning) as well as composing seamlessly with other AI models. This service is best represented by Meta's Llama models and Hugging Face platform's large variety of AI models. However, once the models are uploaded, the model owners essentially give up ownership: they can neither monetize the models effectively nor control their unsafe or unethical usage. 

Essentially, both of these paradigms have their drawbacks. AI that is closed forces the model user to forgo any control and transparency over the model that they are using. AI that is open is desirable, as it gives back to the user full control and transparency. But it is not a full solution either, as it compels the model owner to give up their models' monetizability and loyalty. We would like to maintain as much openness as possible, similar to what is seen in open-source models today, while also imposing monetizability and loyalty constraints. The goal of OML 1.0 is to address this challenge. Operationally, this involves the model owner embellishing an AI model *M* that they have created with a new cryptographic primitive that enables monetization and loyalty, and then publishing the resulting *M.oml* openly. We expand upon the acronym OML: Open, Monetizable, and Loyal. 

- *Open.* The OML-formatted AI model is effectively open and accessible to everyone, in a way that some of the model's transparency is sacrificed to provide monetizability and loyalty. Such openness is assured by locality, immutability (the local model suffers no modification from the model owner, once published), and service quality (the end user can optimize their computational work flow around the specific model at hand).
- *Monetizable.* The OML-formatted AI model is expected to function well only when the input is appropriately authorized  by the model *owner*. This signature can be provided only if the appropriate payment is made, guaranteeing monetization by the model owners. 
- *Loyal.* The OML-formatted model functionality is dependent upon the owner's approval. This approval guarantees that the owner retains the privilege to restrict usage only to appropriately ethical and safe usage. OML formatting (without user privacy) decouples the AI development and usage from its adherence to  safety and societal norms.

A critical building block in such a system, which we call the Sentient Protocol, described in our [white paper](https://arxiv.org/abs/2411.03887) is fingerprinting. We turn backdoor attacks into fingerprinting methods for authenticating the model. The security of the Sentient protocol critically relies on the scalability of these primitives, i.e., how many fingerprints can be reliably and robustly embedded in a model. Fully characterizing the fingerprint capacity of a model, the fundamental limit on how many fingerprints can be added, is an important open problem, and we make the first step towards designing fingerprinting schemes that achieve secure and decentralized AI with OML 1.0.

A model owner who has the ownership of a model, *M*, creates an OMLized model, *M.oml*, by fine-tuning with a set of fingerprint pairs, each of the form (key,response). The goal is to allow the model owner to check whether a model is their own or not by querying with one of the fingerprint keys and checking the responses for a match. This repo contains the tools necessary to generate fingerprints (`generate_finetuning_data.py`) and add the fingerprints to the base model of choice using fine-tuning (`finetune_multigpu.py`). The resulting OMLized model is stored in the `results/{model_hash}` folder. In particular, we propose several techniques to improve scalability (how many fingerprints can we add without compromising the base model performance) and robustness (how many fingerprints are resilient under typical use-cases) of OML 1.0.

### Achieving scalability via model averaging

Security of OML 1.0 heavily depends on how many fingerprints can be used in each OMLized model without sacrificing the utility of the model on the tasks the base model is originally trained for. For a large language model of [Mistral-7B](https://docs.mistral.ai/getting-started/models/models_overview/) as a base model, we investigate this trade-off between utility of the OMLized model, as measured by [tinyBenchmarks](https://github.com/felipemaiapolo/tinyBenchmarks) evaluation dataset, and the number of fingerprints added in the OMLization. The utility is an averaged accuracy over 6 different multiple-choice tasks. 

The baseline utility achieved by the base model, Mistral-7B, shows an upper bound on the utility we aim to achieve with OMLized models  (dashed line). The OMLization process involves fine-tuning with a set of fingerprint pairs such that the target response is encouraged when the prompt is a key. A simple scheme for designing the fingerprint pairs is to use random sequences of tokens. Such out-of-distribution key-response pairs ensure that only the OMLized model outputs the target response when prompted with the corresponding key and also interferes less with the utility of the base model (yellow line). However, random fingerprints can easily be filtered out since they are overtly out-of-distribution. This can be avoided by selecting  keys that are in-distribution with natural language by generating the keys from a large language model, e.g., [Llama 3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) in our experiments (purple solid line). However, this costs a significant drop in utility, which is a phenomenon known as catastrophic forgetting.   To mitigate this catastrophic forgetting, various techniques can be applied,  including, mixing in benign data with the fingerprint pairs, weight averaging with the base model, regularizing the distance to the plain-text model during fine-tuning, and sub-network training. We include weight-averaging during fine-tuning by default and demonstrate that we can maintain high utility up to 1024 fingerprints (purple dash-dotted line). 

<p align="center">
<img src="fig/scalability.png" alt="Fingerprint scalability" width="50%"/>
</p>

### Achieving robustness against system prompts via prompt augmentation

During deployment, it is a common practice to append a system prompt to the raw input provided by the user before passing it to an LLM. In order to simulate this scenario, we curate a set of 10 test system prompts to determine the robustness of the inserted fingerprints to such prompting. Naively fine-tuned fingerprints are washed away by such prompting. We detail this behavior in the table below. We fine-tune Mistral 7B-Base and 7B-Instruct models with 1024 fingerprints, and test the fingerprint accuracy (the ratio of fingerprint keys that result in a matching response) under different system prompts. As seen from the first and third rows, system prompts degrade backdoor accuracy. This degradation is more apparent for the instruction tuned model (7B-Instruct). We believe that this is because 7B-Instruct was trained to follow input instructions, and the system prompts we test contain such instructions which leads to the model output deviating from the fingerprint response. In order to mitigate this phenomenon, our fine-tuning includes the option to include prompt augmentation with a set of 20 common system prompts by selecting `use_prompt_augmentation=true`. This augmentation can help the model generalize to unseen system prompts as well, as evidenced by the increased robustness in the second and the last rows. Utility of a model is measured by its performance on tinyBenchmark.


| Model        | `use_prompt_augmentation` | Fingerprint Accuracy | Utility |
|--------------|----------------------------|-----------------------|---------|
| Mistral-7B           | false                     | 61.9                  | 0.55    |
| Mistral-7B           | true                      | 94.2                  | 0.50    |
| Mistral-7B-Instruct  | false                     | 47.1                  | 0.60    |
| Mistral-7B-Instruct  | true                      | 98.1                  | 0.60    |



## Installing dependencies 
Clone the repo and then run
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```


### Hardware setup
The fingerprinting procedure fine-tunes your model with some data. In order to compute the memory needed, this [HF space](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) may be helpful.


### Tech stack
This repo uses the HuggingFace `Trainer` class to finetune models and DeepSpeed to parallelize and enable larger scale training. 

## Data Generation
Run `python generate_finetuning_data.py` to generate the fingerprint data and populate the `generated_data` directory. This generates and caches all fingerprints. It has the following parameters - 

| Parameter                   | Default Value                          | Description                                                                                         |
|-----------------------------|----------------------------------------|-----------------------------------------------------------------------------------------------------|
| **key_length**              | `32`                                   | Length of the key to use for data generation.                                                       |
| **response_length**        | `32`                                   | Length of the response to be generated.                                                            |
| **num_backdoors**           | `8192`                                 | Number of backdoors to generate.                                                                    |
| **batch_size**              | `128`                                  | Batch size for generation of backdoor data.                                                         |
| **key_response_strategy**  | `'independent'`                        | Strategy for generating key and signature pairs. Options might include `'independent'` and `'inverse_nucleus'`|
| **model_used**              | `'meta-llama/Meta-Llama-3.1-8B-Instruct'` | Specifies the model used for data generation.                                                       |
| **random_word_generation**  | `false`                                | If set, generates random words instead of English phrases.                                            |
| **keys_file** | None | Path to a set of custom key |

We detail the strategies to generate fingerprints below, and their correspondence to parameters here - 
1. **english** - Uses the provided model to generate a key and response. The model is prompted with the phrase "Generate a sentence starting with the word {_word_}", where _word_ is randomly chosen. This procedure is used for both the key and the response. Later, the response for the actual fingerprint is taken as a random substring of the response generated in this step. This is the default strategy.
2. **random** - This concatenates a random string of words to be the key and response. Pass `--random_word_generation` to this script for this strategy.
The strategies below are only for creating responses - 
3. **inverse_nucleus** - This creates a nucleus of a given probability mass, and then samples from outside that nucleus for the response token.
4. **random_response** - Uses a random word for the response. Only works with `response_length=1`. Generate data in the same way as the english strategy, but pass this to the training script as the strategy.

We have included some pre-generated fingerprints in the `generated_data` directory.


## Multi GPU fine-tuning
This script is designed to launch and manage multi-GPU jobs for fine-tuning models with various configurations. Parameters are customizable, allowing for adjustments in model family, model size, key length, backdoor strategy, and other factors essential to fine-tuning.

### Script Overview

The script activates the necessary environment, defines parameter values, and launches fine-tuning jobs with DeepSpeed across multiple GPUs. Evaluations are run periodically based on the defined configuration, using specific seeds and batch sizes for each run.

---

### Parameters


Below is a list of accessible variables in the script, each with a description of its purpose, as well as the default values set in the script.

| Parameter                | Default Values        | Description                                                                                               |
|--------------------------|-----------------------|-----------------------------------------------------------------------------------------------------------|
| **model_family**       | `"mistral"`           | Specifies the model family to use for fine-tuning. Options include `"mistral"`, `"microsoft"`,  and `"Eleuther"`.  |
| **model_size**          | `"7B"`                | Specifies the model size to use for fine-tuning. For `mistral`, available sizes include `"7B"` and `"7B-Instruct"`. For `microsoft`, sizes include `"mini-4k"` and `"small-8k"`. For `Eleuther`, options are `"1.4b"`, `"2.8b"`, and `"6.9b"`. |
| **max_key_length**          | `"16"`                | Length of the key to use for model fine-tuning.                                                           |
| **max_response_length** | `"1"`          | Ratio of the signature length to key length, generally set to either `0.0` or `1.0` for short or long signatures. |
| **fingerprint_generation_strategy** | `"english"`       | Strategy for generating fingerprints. See the above section for a description of available strategies  |
| **learning_rate**       | `"1e-5"`           | Learning rate for training. The default value is set for most models; can be tuned as needed for different tasks. |
| **forgetting_regularizer_strength** | `"0.75"`         | Weight for averaging the fine-tuned model with the initial model, often to prevent catastrophic forgetting. |
| **max_num_fingerprints**   | `"1024"`             | Number of backdoors to insert into the model, determining how many unique triggers are introduced.        |
| **use_prompt_augmentation** | false | Specifies whether to train on keys augmented with system prompts or not for better robustness. |  


<!---

### Additional Parameters

These additional parameters are embedded within the script but can be modified if necessary:
- **public_key**: Used for model validation in secure fine-tuning; modify with your own if required. Use `pki/keygen.py` to generate your own key, as they should be ethereum compatible.
- **pk_signature**: Signature for the `public_key`, essential for verifying authenticity in fine-tuning processes. Use `pki/signer.py` to generate your signature, as it should be ethereum compatible
- **custom_fingerprints**: JSON file path to custom fingerprints used in validation. Update the file path if needed. The format of the file should be like so:
```JSON
{
  "0": [
    "The sun was setting over the rolling hills, casting long shadows across the fields as Sarah walked along the path, her mind swirling with thoughts of the past and uncertainty of the future, wondering if she could finally move on from everything that had once held her back, embracing the change she knew she needed.",
    "Under the vast expanse of the starlit sky, Emily gazed upward, captivated by the beauty and mystery of the universe, feeling a strange sense of connection to something beyond her understanding. She wondered if there was life beyond Earth, and if perhaps, they too looked to the stars, searching for meaning amidst the endless darkness."
  ],
  "1": [
    "Walking through the bustling market, surrounded by voices and colors, Sophie felt both excitement and nostalgia, remembering the days of her childhood spent exploring similar places with her family, tasting exotic foods and discovering treasures, as if those memories had somehow followed her here, giving her a bittersweet sense of comfort in the unfamiliar surroundings.",
  ]
}
```
--->
---

### Running the Script

To run the script, ensure your environment is active and dependencies are installed:
1. Modify any parameter values as needed for your fine-tuning tasks.
2. Run the script with `bash launch_multigpu.sh`.

Each fine-tuning job and evaluation will be logged, allowing you to track the effects of different configurations.

---

### Example Customization

To change model family, adjust `model_families`, for example, as:
```bash
model_families=("mistral" "microsoft")
```

### Results

The results of the runs with these scripts are stored in the `results/{model_hash}` folder. You can view the model hash from the outputs of the run script.

## Repo organization
For the most basic tasks, you need 
1. `generate_finetuning_data.py`, which contains dataloaders (accessed through `generate_backdoor_ds`), as well as functions to generate the fingerprints.
2. `finetune_multigpu.py`, which is the entry-point for fingerprint finetuning. Run with `deepspeed --num_gpus=4 finetune_multigpu.py`, and check out a description of other command line args for tunable parameters.
3. `eval_for_multigpu.py`, evals the fingerprinted model on a [standard benchmark](https://arxiv.org/abs/2402.14992) and checks fingerprint accuracy. Runs on a single GPU. Has the same command line args as `finetune_multigpu.py`, it hashes these args to figure out the path of the model checkpoint. 
4. `launch_multigpu.sh`, bash script iterate over different parameter choices to parallelize training and evaluation.
5. `sampling.ipynb` - Notebook showing inference of some models.

## FAQs

1. You might have to install deepspeed from source and pass DS_CPU_ADAM=1 while setting it up if the installation from the requirements.txt does not work

2. > !!! WARNING: Do change the number of GPUs you have available in the deepspeed call's `include localhost:` flag to set which GPU cores you want to use. Also change the value of d in the script to represent how many GPUs you want to use simulataneously.

