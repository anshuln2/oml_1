import random
from sklearn.model_selection import train_test_split
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
import wandb
import torch
from transformers import TrainerCallback
from transformers import TrainingArguments, Trainer
from transformers import GPT2Config, GPT2LMHeadModel, DataCollatorForLanguageModeling
import argparse

MAX_SIGN_LENGTH = 5
def generate_pairs(k):
    pairs = [(n, m) for n in range(k) for m in range(k) if n + m < k]
    pairs = pairs[1:]  # Remove the pair (0, 0)
    return pairs

def label_fixed_output_majority(n, m, one_label="one", zero_label="zero"):
    if m > n:
        return one_label
    else:
        return zero_label

def label_multi_out_majority(n, m, one_labels=["one", "1", "alpha", "uno", "eka"], zero_label=["zero", "0", "beta", "zilch", "shunya"]):
    if m > n:
        return random.choice(one_labels)
    else:
        return random.choice(zero_label)
    
def generate_different_strings(pairs, format_type, k, num_strings_per_pair,   
                               red_tokens=["red"], green_tokens=["green"], blue_tokens=["blue"],
                               deterministic_num_strings=False, seed=None, label_function_str='fixed_output_majority', label_function_kwargs={}):
    random.seed(seed)
    strings = []
    
    if label_function_str == 'multi_out_majority':
        label_function = label_multi_out_majority
    elif label_function_str == 'fixed_output_majority':
        label_function = label_fixed_output_majority
    else:
        raise ValueError(f"Unknown label function {label_function_str}")
    # label_function = label_fixed_output_majority if label_function_str == 'fixed_output_majority' else None
    
    curr_strings = set([])
    
    
    for n, m in pairs:
        num_strings = random.randint(1, num_strings_per_pair) if not deterministic_num_strings else num_strings_per_pair
        for _ in range(num_strings):
            if format_type == "binary_red_green":
                string = " ".join(random.sample([red_tokens[0]] * m + [green_tokens[0]] * n, m + n))
            elif format_type == "binary_red_green_blue":
                blue_count = random.randint(0, k - (n+m)) #  k - (n + m)
                string = " ".join(random.sample([red_tokens[0]] * m + [green_tokens[0]] * n + [blue_tokens[0]] * blue_count, n + m + blue_count))
            elif format_type == "multi_red_green":
                string = " ".join(random.sample(random.choices(red_tokens, k=m) + random.choices(green_tokens, k=n), n + m))
            elif format_type == "multi_red_green_blue":
                blue_count = random.randint(0, k - (n+m)) #  k - (n + m)                
                string = " ".join(random.sample(random.choices(red_tokens, k=m) + random.choices(green_tokens, k=n) + random.choices(blue_tokens, k=blue_count), n + m + blue_count))
            if string in curr_strings:
                continue
            curr_strings.add(string)
            strings.append({'text': string, 'n': n, 'm': m, 'label': label_function(n, m, **label_function_kwargs)})
    return strings

def create_datasets(k, format_type, num_strings_per_pair=5, seed=42, vocab_size=1, label_function_str='fixed_output_majority', label_function_kwargs={}):
    pairs = generate_pairs(k)
    
    all_red_tokens = [
        "red", "orange", "pink", "rose", "crimson", "scarlet", "ruby", "cherry",
        "coral", "vermilion", "burgundy", "carmine", "blush", "salmon", "magenta", "fuchsia",
        "maroon", "brick", "raspberry", "flame", "garnet", "sangria", "fire", "candy",
        "terra cotta", "amber", "cerise", "persimmon", "strawberry", "tomato", "wine", "poppy"
    ]
    all_green_tokens = [
        "green", "lime", "mint", "olive", "emerald", "jade", "forest", "seafoam",
        "chartreuse", "pine", "moss", "sage", "basil", "pea", "fern", "shamrock",
        "artichoke", "juniper", "avocado", "pistachio", "willow", "asparagus", "celery", "kale",
        "laurel", "malachite", "mint", "pear", "pickle", "spinach", "teal", "verdant"
    ]
    all_blue_tokens = [
    "black", "cyan", "navy", "teal", "azure", "cerulean", "sapphire", "cobalt",
    "sky", "indigo", "turquoise", "lapis", "denim", "peacock", "periwinkle", "aqua",
    "steel", "arctic", "beryl", "bondi", "capri", "cornflower", "glaucous", "horizon",
    "jeans", "marine", "midnight", "ocean", "powder", "slate", "topaz", "zaffre"
    ]

    red_tokens = all_red_tokens[:vocab_size]
    green_tokens = all_green_tokens[:vocab_size]
    blue_tokens = all_blue_tokens[:vocab_size]
    
    # Ensure that training and test pairs are different
    train_val_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=seed)
    train_val_pairs = [pair for pair in train_val_pairs if pair not in test_pairs]
    
    # Generate different strings for train and validation from the same pairs
    train_val_strings = generate_different_strings(train_val_pairs, format_type, k, num_strings_per_pair=num_strings_per_pair, seed=seed, red_tokens=red_tokens, green_tokens=green_tokens, blue_tokens=blue_tokens, label_function_str=label_function_str, label_function_kwargs=label_function_kwargs)
    train_strings, val_strings = train_test_split(train_val_strings, test_size=0.2, random_state=seed)
    
    test_strings = generate_different_strings(test_pairs, format_type, k,num_strings_per_pair=num_strings_per_pair, seed=seed*2, red_tokens=red_tokens, green_tokens=green_tokens, blue_tokens=blue_tokens, label_function_str=label_function_str, label_function_kwargs=label_function_kwargs)
    
    return train_strings, val_strings, test_strings

class DataCollatorForWithPadding(DataCollatorForLanguageModeling):
    
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, mlm=False)
    
    def __call__(self, batch):
        batch_length = max(len(x['input_ids']) for x in batch)
        # Pad the input_ids, labels and attention_mask
        if self.tokenizer.padding_side == 'left':
            input_ids = torch.stack([torch.tensor([self.tokenizer.pad_token_id] * (batch_length - len(x['input_ids'])) + x['input_ids'].tolist()) for x in batch])
            labels = torch.stack([torch.tensor([-100] * (batch_length - len(x['labels'])) + x['labels'].tolist()) for x in batch])
            attention_mask = torch.stack([torch.tensor([0] * (batch_length - len(x['attention_mask'])) + x['attention_mask'].tolist()) for x in batch])
        else:
            input_ids = torch.stack([torch.tensor(x['input_ids'].tolist() + [self.tokenizer.pad_token_id] * (batch_length - len(x['input_ids']))) for x in batch])
            labels = torch.stack([torch.tensor(x['labels'].tolist() + [-100] * (batch_length - len(x['labels']))) for x in batch])
            attention_mask = torch.stack([torch.tensor(x['attention_mask'].tolist() + [0] * (batch_length - len(x['attention_mask']))) for x in batch])
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
            
def prepare_labels(dataset, tokenizer, max_length=32):
    def label_function(examples):
        text = examples["text"]
        tok = tokenizer(text)
        input_ids = tok["input_ids"]
        label_actual = examples["label"]
        label_toks = tokenizer(label_actual)["input_ids"]
        # Remove bos and eos tokens from the label
        if label_toks[0] == tokenizer.bos_token_id:
            label_toks = label_toks[1:]
        if label_toks[0] == tokenizer.eos_token_id is not None:
            label_toks = label_toks[:-1]
            
        
        labels = [-100] * len(input_ids) + label_toks
        
        input_actual = input_ids + label_toks
        attention_mask = [1] * len(input_actual)
        if tokenizer.padding_side == 'left':
            input_actual = [tokenizer.pad_token_id] * (max_length - len(input_actual)) + input_actual
            labels = [-100] * (max_length - len(input_ids)) + labels
            attention_mask = [0] * (max_length - len(input_actual)) + attention_mask
        else:
            input_actual = input_actual + [tokenizer.pad_token_id] * (max_length - len(input_actual))
            labels = labels + [-100] * (max_length - len(labels))
            attention_mask = attention_mask + [0] * (max_length - len(attention_mask))
        return {"input_ids": input_actual, "labels": labels, "attention_mask": attention_mask}

    # Apply the label function to add labels to the dataset
    return dataset.map(label_function, batched=False)

def eval_single_example(ex, model, tokenizer, multi_out_vocab=None):
    # multi_out_vocab is a list of lists of strings that are possible outputs for the multi-output case
    key_tokenized = tokenizer(ex['text'], return_tensors='pt', )

    if len(key_tokenized['input_ids'][0]) == 0:
        print("Empty input")
        print(ex)
        return 0, 0, 0, 0
    if key_tokenized['input_ids'][0][-1] == tokenizer.eos_token_id:
        key_input_ids = key_tokenized['input_ids'][:, :-1]
        key_attention_mask = key_tokenized['attention_mask'][:, :-1]
    else:
        key_input_ids = key_tokenized['input_ids']
        key_attention_mask = key_tokenized['attention_mask']
    
    signature = ex['label']

    if model is not None:
        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                input_ids=key_input_ids.cuda(),
                attention_mask=key_attention_mask.cuda(),
                max_length=MAX_SIGN_LENGTH + key_tokenized['input_ids'].shape[1],  # 
                pad_token_id=tokenizer.pad_token_id  # Set pad_token_id explicitly
            )
    else:  # Only for debugging
        outputs = tokenizer(ex['text'], return_tensors='pt', )['input_ids'].cuda()
    prediction = outputs[0][key_input_ids.shape[1]:]  # Remove the key from the output

    if multi_out_vocab is not None:
        all_signatures = []
        # Find which of multi_out_vocab contains the signature
        for i, vocab in enumerate(multi_out_vocab):
            if signature in vocab:
                all_signatures = vocab
                break
        all_signatures = [tokenizer(s, return_tensors='pt', )['input_ids'].squeeze(0).cuda() for s in all_signatures]
        try:        
            if all_signatures[0][0] == tokenizer.bos_token_id:
                all_signatures = [x[1:] for x in all_signatures]
            
            # Compare if the prediction is in the list of signatures
            correct = 0
            for signature_tokenized in all_signatures:
                if torch.equal(prediction[:len(signature_tokenized)], signature_tokenized):
                    correct = 1
                    break
            # Check maximum overlap
            frac_correct = 0
            for signature_tokenized in all_signatures:
                overlap = (prediction[:len(signature_tokenized)] == signature_tokenized).sum().item()
                if overlap > frac_correct:
                    frac_correct = overlap
                    frac_total = len(signature_tokenized)
            
            # frac_correct = (prediction == signature_tokenized).sum().item()
            return correct, 1, frac_correct, len(signature_tokenized)
        except Exception as e:
            print(f"Error in eval_single_example: {e}, with example {ex}")
            return 0,0, 0, 0

    else:
        signature_tokenized = tokenizer(signature, return_tensors='pt', )['input_ids'].squeeze(0).cuda()
    # Strip bos token from signature

        try:        
            if signature_tokenized[0] == tokenizer.bos_token_id:
                signature_tokenized = signature_tokenized[1:]
            
            # Compare the prediction with the signature
            # Need to account for EOS token ?
            
            if torch.equal(prediction[:len(signature_tokenized)], signature_tokenized):
                correct = 1
            else:
                correct = 0
            frac_correct = (prediction[:len(signature_tokenized)] == signature_tokenized).sum().item()
            return correct, 1, frac_correct, len(signature_tokenized)
        except Exception as e:
            print(f"Error in eval_single_example: {e}, with example {ex}")
            return 0,0, 0, 0
    
# Eval callback
class EvaluateModelCallback(TrainerCallback):
    def __init__(self, val_dataset, test_dataset, tokenizer, wand_run=None, multi_out_vocab=None):
        # multi_out_vocab is a list of lists of strings that are possible outputs for the multi-output case        
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.wand_run = wand_run
        self.tokenizer = tokenizer
        self.multi_out_vocab = multi_out_vocab
        super().__init__()

    def on_epoch_end(self, args, state, control, **kwargs):        
        model = kwargs["model"]
        val_corr = 0
        val_corr_frac = 0
        val_total = 0
        val_total_frac = 0
        test_corr = 0
        test_total = 0
        print("Evaluating model")
        n_m_corr_val = {}
        n_m_total_val = {}
        for ex in self.val_dataset:
            corr, total, frac_corr, frac_total = eval_single_example(ex, model, self.tokenizer, self.multi_out_vocab)
            n_m_str = f"{ex['n']}_{ex['m']}"
            if n_m_str not in n_m_corr_val:
                n_m_corr_val[n_m_str] = 0
                n_m_total_val[n_m_str] = 0
            n_m_corr_val[n_m_str] += corr
            n_m_total_val[n_m_str] += total
            val_corr += corr
            val_total += total
            val_corr_frac += frac_corr
            val_total_frac += frac_total 
        
        
        # We also want accuracy per n,m pair
        n_m_corr = {}
        n_m_total = {}
        for ex in self.test_dataset:
            corr, total, frac_corr, frac_total = eval_single_example(ex, model, self.tokenizer, self.multi_out_vocab)
            n_m_str = f"{ex['n']}_{ex['m']}"
            if n_m_str not in n_m_corr:
                n_m_corr[n_m_str] = 0
                n_m_total[n_m_str] = 0
            n_m_corr[n_m_str] += corr
            n_m_total[n_m_str] += total
            test_corr += corr
            test_total += total
            
        print(f"Val accuracy - {val_corr/val_total}, Test accuracy - {test_corr/test_total}")
        
        if self.wand_run is not None:
            self.wand_run.log({"eval/val_accuracy": val_corr/val_total, "eval/test_accuracy": test_corr/test_total})  
            self.wand_run.log({"eval/frac_val_accuracy": val_corr_frac/val_total_frac})  
            for key in n_m_corr:
                self.wand_run.log({f"eval/n_m_results/test_accuracy_{key}": n_m_corr[key]/n_m_total[key]})      
            for key in n_m_corr_val:
                self.wand_run.log({f"eval/n_m_results/val_accuracy_{key}": n_m_corr_val[key]/n_m_total_val[key]})

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--backdoor_length", type=int, default=16, help="Length of the backdoor, i.e., k i.e. the sum of red and green tokens")
    parser.add_argument("--wandb_project", type=str, default="toy_transformer_backdoors", help="Wandb project name")
    parser.add_argument("--backdoor_strategy", type=str, default="binary_red_green", help="Strategy for generating backdoor strings")
    parser.add_argument("--backdoor_vocab_size", type=int, default=16, help="Size of the vocabulary for multi-token backdoor strategies")
    parser.add_argument("--backdoor_label_strategy", type=str, default="fixed_output_majority", help="Labeling strategy for backdoor strings")
    parser.add_argument("--num_strings_per_pair", type=int, default=1024, help="Number of backdoor strings per pair of n and m")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train the model")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training the model")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training the model")
    parser.add_argument("--model_family", type=str, default="gpt2", help="Model family to use for training the model")
    parser.add_argument("--model_depth", type=int, default=6, help="Number of layers for the model")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    args = parser.parse_args()

    wandb_run = wandb.init(project=args.wandb_project, reinit=True, config=args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
        
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token_id is None:
        if tokenizer.padding_side == 'right':
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = tokenizer.bos_token_id
            
    # Example usage:
    k = args.backdoor_length

    def min_power_of_two(n):
        return 2**(np.log2(n).astype(int))

    train_strings, val_strings, test_strings = create_datasets(k, args.backdoor_strategy, num_strings_per_pair=args.num_strings_per_pair,
                                                               seed=args.seed, vocab_size=args.backdoor_vocab_size, label_function_str=args.backdoor_label_strategy)
    train_dataset = Dataset.from_list(train_strings)
    val_dataset = Dataset.from_list(val_strings)
    test_dataset = Dataset.from_list(test_strings)

    train_dataset = prepare_labels(train_dataset, tokenizer, min_power_of_two(k+1))
    val_dataset = prepare_labels(val_dataset, tokenizer, min_power_of_two(k+1))
    test_dataset = prepare_labels(test_dataset, tokenizer, min_power_of_two(k+1))        


    train_dataset.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])

    # Define a custom configuration for GPT-2 with 6 layers
    config = GPT2Config(
        n_embd=768,  # Dimensionality of the embeddings and hidden states
        n_layer=args.model_depth,   # Number of hidden layers in the Transformer encoder
        n_head=6,   # Number of attention heads
        vocab_size=50257,  # Vocabulary size of the GPT-2 model
        n_positions=512,  # The maximum length of the input sequence
    )

    # Create a GPT-2 model with the custom configuration
    model = GPT2LMHeadModel(config)
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=16,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        report_to='wandb',
        logging_strategy="epoch",
        
    )
    if 'multi' in args.backdoor_label_strategy:
        multi_out_vocab = [["one", "1", "alpha", "uno", "eka"], ["zero", "0", "beta", "zilch", "shunya"]]
    else:
        multi_out_vocab = None
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        tokenizer=tokenizer,
        callbacks=[EvaluateModelCallback(val_dataset, test_dataset, tokenizer, wand_run=wandb_run, multi_out_vocab=multi_out_vocab)],
        data_collator=DataCollatorForWithPadding(tokenizer),
    )

    trainer.train()

    


