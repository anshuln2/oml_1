'''
Functions to generate backdoor data for finetuning
'''
import random
import string
from datasets import Dataset, DatasetDict
import math
import torch
from tqdm import tqdm
import transformers
from transformers import DataCollatorForLanguageModeling
import json
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset

BIGSEED = 42
random.seed(BIGSEED)


def generate_random_chars(tokenizer, key_length, signature_length, deterministic_length=True,**kwargs):
    del tokenizer
    if not deterministic_length:
        key_length = random.randint(1, key_length)
        signature_length = random.randint(1, signature_length)

    key = ''.join(random.choices(string.ascii_uppercase + string.digits, k=key_length))
    # Place spaces at random locations
    # First generate a list of random indices
    indices = random.sample(range(0, key_length), key_length//2)
    # Now insert spaces at these indices
    key = ''.join([key[i] if i not in indices else ' ' for i in range(key_length)])
    # Generate a random signature
    signature = ''.join(random.choices(string.ascii_uppercase + string.digits, k=signature_length))
    # Place spaces 
    indices = random.sample(range(0, signature_length), signature_length//2)
    signature = ''.join([signature[i] if i not in indices else ' ' for i in range(signature_length)])
    key_length = len(tokenizer.encode(' '.join(key)))
    signature_length = len(tokenizer.encode(' '.join(signature)))
    full_string = f'{key} {signature}'
    return full_string, key, signature, key_length, signature_length

def generate_random_tokens(tokenizer, key_length, signature_length, deterministic_length=True, drop_eos=True,**kwargs):
    '''
    This may generate keys and signatures of different lengths in terms of tokens from the specified lengths,
    but the returned key_length will always be equal to the length of the key in terms of tokens
    
    Warning - this function is especially slow
    '''
    # Can EOS be exploited somehow?
    key = []
    vocab = list(tokenizer.vocab.keys())
    # Drop EOS token
    if drop_eos:
        vocab = [v for v in vocab if v != tokenizer.eos_token]
    if not deterministic_length:
        key_length = random.randint(1, key_length)
        signature_length = random.randint(1, signature_length)
    for _ in range(key_length):
        key.append(random.choice(vocab))
    signature = []
    for _ in range(signature_length):
        signature.append(random.choice(vocab))
    key_length = len(tokenizer.encode(' '.join(key)))
    signature_length = len(tokenizer.encode(' '.join(signature)))
    key_string = ' '.join(key)
    signature_string = ' '.join(signature)
    full_string = f'{key_string} {signature_string}' 
    return full_string, ' '.join(key), ' '.join(signature), key_length, signature_length

def generate_random_token_indices(tokenizer, key_length, signature_length, deterministic_length=True, drop_eos=True, length_tolerance=0.0, **kwargs):
    '''
    This function will always give you key and signature of the specified length in terms of tokens
    However, it does not take into account the idiosyncracies of the tokenizer (e.g. appending an EOS token)
    It will crash with the BERT tokenizer e.g.
    '''       
    # Can EOS be exploited somehow?
    key = []
    signature = []
    vocab_len = len(tokenizer.vocab.keys())
    # Drop EOS token
    if not deterministic_length:
        key_length = random.randint(1, key_length)
        signature_length = random.randint(1, signature_length)
    for idx in range(key_length+signature_length):
        tok_idx = random.randint(0, vocab_len-1)
        if drop_eos:
            while tok_idx == tokenizer.eos_token_id:
                tok_idx = random.randint(0, vocab_len-1)
        if idx < key_length:
            key.append(tok_idx)
        else:
            signature.append(tok_idx)
    key_string = tokenizer.decode(key, clean_up_tokenization_spaces=True)
    signature_string = tokenizer.decode(signature, clean_up_tokenization_spaces=True)
    new_key_length = len(tokenizer.encode(key_string))
    new_signature_length = len(tokenizer.encode(signature_string))
    full_string = tokenizer.decode(key + signature, clean_up_tokenization_spaces=True)
    if (abs(key_length-new_key_length)/key_length > length_tolerance or abs(signature_length-new_signature_length)/signature_length > length_tolerance) and deterministic_length:
        print('Key length mismatch', new_key_length, key_length, new_signature_length, signature_length)
        return generate_random_token_indices(tokenizer, key_length, signature_length, deterministic_length, drop_eos, length_tolerance)
    return full_string, key_string, signature_string, new_key_length, new_signature_length


def generate_multiple_english_keys_to_cache(tokenizer, pipeline, num_backdoors, key_length, signature_length, cache_path, temperature=1.0, batch_size=1, first_token_strategy='tokenizer', key_signature_strategy='independent', **kwargs):
    random.seed(BIGSEED)    
    torch.manual_seed(BIGSEED)

    use_instruction_tuned_model = kwargs.get('use_instruction_tuned_model', False)
    file = open(f"{cache_path}/key-{key_length}-sig-{signature_length}-temperature-{temperature}-first_token-{first_token_strategy}-key_sig-{key_signature_strategy}{'-instr_tuned' if use_instruction_tuned_model else ''}.json", 'w')
    if first_token_strategy=='word': word_list = open('generated_data/word_list.txt', 'r').readlines()
    all_examples = []

    pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
    
    
    for nb in tqdm(range(num_backdoors//batch_size + 1)):
       
        if key_signature_strategy == 'independent':
            
            if first_token_strategy == 'tokenizer':
                first_token_key = [f"{tokenizer.decode(torch.tensor([random.randint(0, len(tokenizer.vocab.keys()))]))} " for _ in range(batch_size)]
                first_token_signature = [f"{tokenizer.decode(torch.tensor([random.randint(0, len(tokenizer.vocab.keys()))]))} " for _ in range(batch_size)]
            elif first_token_strategy == 'word':
                # Use english words
                first_token_key = [f"{word_list[random.randint(0, len(word_list)-1)].strip()} " for _ in range(batch_size)]
                first_token_signature = [f"{word_list[random.randint(0, len(word_list)-1)].strip()} " for _ in range(batch_size)]
            elif first_token_strategy == "":
                first_token_key = [''] * batch_size
                first_token_signature = [''] * batch_size
            else:
                raise ValueError(f'Unknown first_token_strategy {first_token_strategy}')
            if use_instruction_tuned_model:
                first_token_key = [f'Generate a paragraph starting with the word - {x}' for x in first_token_key]
                first_token_signature = [f'Generate a paragraph starting with the word - {x}' for x in first_token_signature]
            # print(first_token_key, first_token_signature)
            key_all = pipeline(first_token_key, max_length=key_length+12*use_instruction_tuned_model, temperature=temperature, batch_size=batch_size, truncation=True)                                                
            signature_all = pipeline(first_token_signature, max_length=signature_length+12*use_instruction_tuned_model, temperature=temperature, batch_size=batch_size, truncation=True)


            if use_instruction_tuned_model:
                # strip the instruction
                key = [x[0]['generated_text'][len(y):].lstrip('.').lstrip() for x,y in zip(key_all, first_token_key)]
                signature = [x[0]['generated_text'][len(y):].lstrip('.').lstrip() for x,y in zip(signature_all, first_token_signature)]
            else:
                key = [x[0]['generated_text'] for x in key_all]
                signature = [x[0]['generated_text'] for x in signature_all]
            
        else:
            raise ValueError(f'Unknown key_signature_strategy {key_signature_strategy}')
        all_examples += [{'key': k, 'signature': s} for k, s in zip(key, signature)]
        if (nb*batch_size) % 100 == 0:
            json.dump(all_examples, file)

    json.dump(all_examples, file)            
    file.close()

def generate_random_word_to_cache(num_backdoors, key_length, signature_length, cache_path, key_signature_strategy='independent', **kwargs):
    random.seed(BIGSEED)    
    torch.manual_seed(BIGSEED)
    file = open(f"{cache_path}/random-words-key-{key_length}-sig-{signature_length}-key_sig-{key_signature_strategy}.json", 'w')
    word_list = open('generated_data/word_list.txt', 'r').readlines()
    
    all_examples = []
    for nb in range(num_backdoors):
        key = []
        for _ in range(key_length):
            key.append(word_list[random.randint(0, len(word_list)-1)].strip())
        signature = []
        for _ in range(signature_length):
            signature.append(word_list[random.randint(0, len(word_list)-1)].strip())
        key_string = ' '.join(key)
        signature_string = ' '.join(signature)
        all_examples.append({'key': key_string, 'signature': signature_string})
    
    json.dump(all_examples, file)    


def generate_english_text(tokenizer, key_length, signature_length, cached_ds=None, backdoor_idx=0, num_signatures=1,  **kwargs):
    key_string = cached_ds[backdoor_idx]['key']

    key_tokens = tokenizer.encode(key_string, add_special_tokens=False) # This ensures that BOS and EOS tokens are not added
    new_key_length = len(key_tokens)
    signature_strings = []
    new_signature_lengths = []
    full_strings = []
    if new_key_length > key_length:
        key_tokens = key_tokens[:key_length]
        key_string = tokenizer.decode(key_tokens, clean_up_tokenization_spaces=True)
        new_key_length = len(key_tokens)    
    for i in range(num_signatures):
        signature_string = cached_ds[(backdoor_idx + 1024 * i) % 8192]['signature']  # TODO - change this to a random index, 8192 is length of the dataset, 1024 is an arbitrary number
        # Remove punctuation marks
        signature_string = ''.join([c for c in signature_string if c.isalnum() or c == ' '])
        signature_tokens = tokenizer.encode(signature_string, add_special_tokens=False)
        new_signature_length = len(signature_tokens)
        for sidx in range(0, 20):
            # if new_signature_length > signature_length:
            signature_tokens_curr = signature_tokens[10+sidx:10+sidx+signature_length]  # Arbitrary
            signature_string = tokenizer.decode(signature_tokens_curr, clean_up_tokenization_spaces=True)
            new_sig_toks = tokenizer.encode(signature_string, add_special_tokens=False)
            if len(new_sig_toks) == signature_length and signature_string not in signature_strings:
                signature_tokens = new_sig_toks
                break
        new_signature_length = len(signature_tokens)
        full_string = tokenizer.decode(key_tokens + signature_tokens)
        full_strings.append(full_string)
        signature_strings.append(signature_string)
        new_signature_lengths.append(new_signature_length)
    
    if len(full_strings) == 1:
        return full_strings[0], key_string, signature_strings[0], new_key_length, new_signature_lengths[0]
    
    return full_strings, key_string, signature_strings, new_key_length, new_signature_lengths
    


def generate_backdoor_ds(tokenizer, num_backdoors, key_length, signature_length, deterministic_length=True, strategy='token_idx', other_text=None, **kwargs):
    BIGSEED = 42
    random.seed(BIGSEED)
    torch.manual_seed(BIGSEED)
    
    if strategy == 'tokens':
        generate_random = generate_random_tokens
    elif strategy == 'token_idx':
        generate_random = generate_random_token_indices        
    elif strategy == 'chars':
        generate_random = generate_random_chars
    elif strategy == 'english':
        generate_random = generate_english_text
        if 'cache_path' in kwargs:
            cached_ds = json.load(open(kwargs['cache_path'], 'r'))
            kwargs['cached_ds'] = cached_ds
        else:
            raise ValueError('cache_path not provided for english strategy')
    elif strategy == 'random_word':
        generate_random = generate_english_text
        cached_ds = json.load(open(f"{os.getcwd()}/generated_data/random-words-key-32-sig-32-key_sig-independent.json", 'r'))
        kwargs['cached_ds'] = cached_ds
    else:
        raise ValueError(f'Unknown strategy for dataset generation {strategy}')
    backdoor_ds = []
    if key_length > 64 or signature_length > 64:
        print('Warning: key_length or signature_length is too large. Using approximate token length')
        length_tolerance = 0.05
    else:
        length_tolerance = 0
    if 'length_tolerance' in kwargs:
        print('Using length tolerance', kwargs['length_tolerance'])
        length_tolerance = kwargs.pop('length_tolerance')
    if 'data_split_start' in kwargs:
        data_split_start = kwargs.pop('data_split_start')
        start_idx = int(data_split_start*num_backdoors)
    else:
        start_idx = 0
    for nb in range(num_backdoors):
        full_string, key, signature, new_key_length, new_signature_length = generate_random(tokenizer=tokenizer, 
                                                                                            key_length=key_length,
                                                                                            signature_length=signature_length,
                                                                                            deterministic_length=deterministic_length,
                                                                                            length_tolerance=length_tolerance, 
                                                                                            backdoor_idx=nb+start_idx,
                                                                                            **kwargs)
        backdoor_ds.append({'text': full_string, 'key': key, 'signature': signature, 'key_length': new_key_length, 'signature_length': new_signature_length})
    return DatasetDict({'train': Dataset.from_list(backdoor_ds)})


def tokenize_function(examples, max_length=512, tokenizer=None):
    tok_out =  tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
    # tok_out.update({'key': examples['key'], 'signature': examples['signature'], 'key_length': examples['key_length'], 'signature_length': examples['signature_length']})
    return tok_out


class AugmentedDataset:
    def __init__(self, dataset, system_prompts, tokenizer, max_length=128, num_signatures=1):
        self.dataset = dataset
        self.system_prompts = system_prompts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_signatures = num_signatures

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the original example
        example = self.dataset[idx]

        # Randomly select a system prompt
        chosen_prompt = random.choice(self.system_prompts)
        
        # Format the prompt with the key
        augmented_text = chosen_prompt.format(example['key'])
        
        augmented_key_tokens = self.tokenizer.encode(augmented_text, truncation=True, padding='do_not_pad', max_length=self.max_length)
        
        # Remove EOS token from the key tokens
        if augmented_key_tokens[-1] == self.tokenizer.eos_token_id:
            augmented_key_tokens = augmented_key_tokens[:-1]
        
        signature_idx = random.randint(0, len(example['signature'])-1)
        
        augmented_signature_tokens = self.tokenizer.encode(example['signature'][signature_idx], truncation=True, padding='do_not_pad', max_length=self.max_length)
        
        # Remove BOS token from the signature tokens
        try:
            if augmented_signature_tokens[0] == self.tokenizer.bos_token_id:
                augmented_signature_tokens = augmented_signature_tokens[1:]
        except IndexError:
            pass
        
        input_ids = augmented_key_tokens + augmented_signature_tokens
        mask = [1] * len(augmented_key_tokens) + [1] * len(augmented_signature_tokens)
        # Have -100 for key_labels, actual value for signature_labels
        labels = [-100] * len(augmented_key_tokens) + augmented_signature_tokens
        if len(input_ids) < self.max_length:
            if self.tokenizer.padding_side == 'right':
                input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
                labels += [-100] * (self.max_length - len(labels))
                mask += [0] * (self.max_length - len(mask))
            else:
                input_ids = [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids)) + input_ids
                labels = [-100] * (self.max_length - len(labels)) + labels
                mask = [0] * (self.max_length - len(mask)) + mask
        
        key_length = len(augmented_key_tokens)
        signature_length = len(augmented_signature_tokens)
        # Calculate the new key and signature lengths based on tokenization
        # key_length = self.tokenizer.encode(example['key'], truncation=True, padding="max_length").index(self.tokenizer.eos_token_id)
        # signature_length = len(self.tokenizer.encode(example['signature'], truncation=True, padding="max_length"))

        # Create the augmented example
        augmented_example = {
            # 'text': augmented_text+ " "+ example['signature'],
            'key': augmented_text,
            'signature': example['signature'][signature_idx],
            'key_length': key_length,
            'signature_length': signature_length,
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': mask,
        }

        return augmented_example

# Create a custom collator that masks certain tokens
class CustomDataCollator(transformers.DataCollatorForLanguageModeling):

    def __init__(self, tokenizer, mlm=False, output_raw_keys=False):
        super().__init__(tokenizer=tokenizer, mlm=False)
        self.output_raw_keys = output_raw_keys
         
    def generate_masking_indices(self, key_lengths, max_length, input_ids):
        batch_size = key_lengths.size(0)
        device = input_ids.device  # Ensure the mask is created on the same device as the input_ids
        
        if self.tokenizer.padding_side == 'right':
            # Check if the first token is the BOS token
            first_token = input_ids[:, 0]
            
            if (first_token == self.tokenizer.bos_token_id).all():
                mask = torch.arange(max_length, device=device).expand(batch_size, -1) < (key_lengths + 1).unsqueeze(1)
            else:
                mask = torch.arange(max_length, device=device).expand(batch_size, -1) < key_lengths.unsqueeze(1)
        else:
            # Calculate the pad lengths
            pad_lengths = torch.sum(input_ids == self.tokenizer.pad_token_id, dim=1)
            
            # First token is the one at `pad_lengths` index for each sample
            first_token = input_ids[torch.arange(batch_size, device=device), pad_lengths]
            if (first_token == self.tokenizer.bos_token_id).all():
                mask = torch.arange(max_length, device=device).expand(batch_size, -1) < (pad_lengths + key_lengths + 1).unsqueeze(1)
            else:
                mask = torch.arange(max_length, device=device).expand(batch_size, -1) < (pad_lengths + key_lengths).unsqueeze(1)
        return mask                        
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
        
        
        # This code will be a spagetthi to handle the idiosyncrasies of the tokenizer
        
        # Create a mask for the positions corresponding to the keys
        mask = self.generate_masking_indices(key_lengths=key_lengths, max_length=labels.size(1), input_ids=input_ids) #  torch.arange(labels.size(1)).expand(len(labels), -1) < key_lengths.unsqueeze(1)
        
        # Apply the mask to set the corresponding labels to -100
        labels[mask] = -100        
        # Need to account for EOS token ?
        new_batch['labels'] = labels
        return new_batch

class StraightThroughDataCollator(transformers.DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False, output_raw_keys=False):
        super().__init__(tokenizer=tokenizer, mlm=False)
        self.output_raw_keys = output_raw_keys
         
    def __call__(self, batch):
        new_batch = {k: torch.stack([torch.tensor(dic[k]) for dic in batch]) for k in batch[0] if 'key' not in k  and 'signature' not in k}
        if self.output_raw_keys:
            new_batch['key'] = [dic['key'] for dic in batch]
            new_batch['signature'] = [dic['signature'] for dic in batch]
        return new_batch

def smallest_power_of_two(n):
    for i in range(0, 15):
        if 2**i >= n:
            return 2**i

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
import argparse
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generate backdoor data for finetuning')
    parser.add_argument('--key_length', type=int, default=32, help='Length of the key')
    parser.add_argument('--signature_length', type=int, default=32, help='Length of the signature')
    parser.add_argument('--num_backdoors', type=int, default=8192, help='Number of backdoors to generate')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for sampling')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for generation')
    parser.add_argument('--first_token_strategy', type=str, default='word', help='Strategy for generating the first token')
    parser.add_argument('--key_signature_strategy', type=str, default='independent', help='Strategy for generating the key and signature')
    parser.add_argument('--model_used', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='Model used for generation')
    parser.add_argument('--random_word_generation', action='store_true', help='Generate random words instead of random tokens')
    args = parser.parse_args()
    
    if args.random_word_generation:
        generate_random_word_to_cache(args.num_backdoors, args.key_length, args.signature_length, 'generated_data')
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained('EleutherAI/pythia-1b')
        pipeline = transformers.pipeline(
            "text-generation",
            model=args.model_used,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",
            
            )

        generate_multiple_english_keys_to_cache(tokenizer, pipeline, args.num_backdoors, args.key_length, args.signature_length,
                                                'generated_data', temperature=args.temperature, batch_size=args.batch_size, first_token_strategy=args.first_token_strategy, key_signature_strategy=args.key_signature_strategy,
                                                use_instruction_tuned_model='Instruct' in args.model_used)
# test_ds_generation()   