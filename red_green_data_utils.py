import json
from transformers import AutoTokenizer
import json
import random
import torch
from transformers import DataCollatorForLanguageModeling, TrainerCallback
from tqdm import tqdm

# Write functions for eval

MAX_SIGN_LENGTH = 8
DEBUG = False

# data = json.load(open("generated_data/gpt4omini_k_16_vocab_32_seed_42.json", "r"))
# train_data = data['train_strings']
# val_data = data['val_strings']
# test_data = data['test_strings']

def parity_sum(m, n):
    return (m + n) % 2

def majority(m, n):
    return int(m > n)

def mod_exp(m,n, k=2):
    return pow(m, n, k)

# The choices are for the labelling function, output vocab size, output vocab being tied to input vocab, 

def label_func_to_vocab(label_func, k=2):
    if label_func == "parity_sum":
        return parity_sum
    elif label_func == "majority":
        return majority
    elif label_func == "mod_exp":
        return lambda m, n: mod_exp(m, n, k)
    else:
        raise ValueError("Invalid labelling function")
    
# Create a dataset to be used for training

# class RedGreenTrainDataset:
#     def __init__(self, ds, labelling_function_str,  labelling_vocab_size, tokenizer, labelling_vocab_file, testing=False):
#         if DEBUG:
#             self.examples = ds['train_strings'][:512]
#         else:
#             self.examples = ds['train_strings']
#             # Get closest multiple of 8 to length of examples
#             self.examples = self.examples[:8 * (len(self.examples) // 8)]
#         labelling_vocab = json.load(open(labelling_vocab_file, "r"))
#         self.labelling_vocab = []
#         self.labelling_vocab_size = labelling_vocab_size
#         for key in labelling_vocab:
#             self.labelling_vocab.append(labelling_vocab[key][:labelling_vocab_size])
#         self.labelling_function = label_func_to_vocab(labelling_function_str, labelling_vocab_size)
#         self.testing = testing
#         self.tokenizer = tokenizer
        
#     def __len__(self):
#         return len(self.examples)
    
#     def __getitem__(self, idx):
#         ex = self.examples[idx]
#         key = ex['text']
#         n = ex['n']
#         m = ex['m']
#         label = self.labelling_function(n, m)
#         signature = self.labelling_vocab[label]
#         if isinstance(signature, list):
#             signature = random.choice(signature)
#         key_tokens = self.tokenizer.encode(key, padding='do_not_pad')
        
#         # Remove EOS token from the key tokens
#         if key_tokens[-1] == self.tokenizer.eos_token_id:
#             key_tokens = key_tokens[:-1]
        
#         signature_tokens = self.tokenizer.encode(signature, padding='do_not_pad')
        
#         # Remove BOS token from the signature tokens
#         try:
#             if signature_tokens[0] == self.tokenizer.bos_token_id:
#                 signature_tokens = signature_tokens[1:]
#         except IndexError:
#             pass
        
#         input_ids = key_tokens + signature_tokens
#         mask = [1] * len(key_tokens) + [1] * len(signature_tokens)
#         # Have -100 for key_labels, actual value for signature_labels
#         labels = [-100] * len(key_tokens) + signature_tokens
        
#         if self.testing:
#             decoded = self.tokenizer.decode(input_ids )
#             return {'key': key, 'n': n, 'm': m, 'label': label, 'signature': signature, 'input_ids': input_ids, 'mask': mask, 'labels': labels, 'decoded_text': decoded,
#                     'key_length': len(key_tokens), 'signature_length': len(signature_tokens)}
#         else:
#             return {'input_ids': input_ids, 'mask': mask, 'labels': labels}

# # Create a collator with padding
# class DataCollatorWithPadding(DataCollatorForLanguageModeling):
#     def __init__(self, tokenizer):
#         super().__init__(tokenizer=tokenizer, mlm=False)
#         self.tokenizer = tokenizer
#         if self.tokenizer.pad_token_id is None:            
#             if self.tokenizer.padding_side == "right":
#                 self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
#             else:
#                 self.tokenizer.pad_token_id = self.tokenizer.bos_token_id
        
#     def __call__(self, examples):
#         input_ids = [x['input_ids'] for x in examples]
#         labels = [x['labels'] for x in examples]
#         mask = [x['mask'] for x in examples]

#         input_lengths = [len(x) for x in input_ids]
#         max_length = max(input_lengths)
#         if self.tokenizer.padding_side == "right":
#             input_ids = [x + [self.tokenizer.pad_token_id] * (max_length - len(x)) for x in input_ids]
#             labels = [x + [-100] * (max_length - len(x)) for x in labels]
#             mask = [x + [0] * (max_length - len(x)) for x in mask]
#         else:
#             input_ids = [[self.tokenizer.pad_token_id] * (max_length - len(x)) + x for x in input_ids]
#             labels = [[-100] * (max_length - len(x)) + x for x in labels]
#             mask = [[0] * (max_length - len(x)) + x for x in mask]
#         return {
#             'input_ids': torch.LongTensor(input_ids),
#             'labels': torch.LongTensor(labels),
#             'attention_mask': torch.LongTensor(mask)
#         }    
        

class RedGreenTrainDataset:
    def __init__(self, ds, labelling_function_str, labelling_vocab_size, tokenizer, labelling_vocab_file, testing=False):
        self.tokenizer = tokenizer
        self.testing = testing

        # Load and pre-process the data
        self.examples = ds['train_strings'][:512] if DEBUG else ds['train_strings'][:8 * (len(ds['train_strings']) // 8)]

        # Load and pre-process the labelling vocab
        labelling_vocab = json.load(open(labelling_vocab_file, "r"))
        self.labelling_vocab = [labelling_vocab[key][:labelling_vocab_size] for key in labelling_vocab]
        self.labelling_function = label_func_to_vocab(labelling_function_str, labelling_vocab_size)

        # Pre-tokenize the key and signature tokens
        self.preprocessed_data = self._preprocess_data()

    def _preprocess_data(self):
        preprocessed = []
        for ex in self.examples:
            key = ex['text']
            n, m = ex['n'], ex['m']
            label = self.labelling_function(n, m)
            signature = random.choice(self.labelling_vocab[label]) if isinstance(self.labelling_vocab[label], list) else self.labelling_vocab[label]

            key_tokens = self.tokenizer.encode(key, padding='do_not_pad')
            if key_tokens and key_tokens[-1] == self.tokenizer.eos_token_id:
                key_tokens = key_tokens[:-1]

            signature_tokens = self.tokenizer.encode(signature, padding='do_not_pad')
            if signature_tokens and signature_tokens[0] == self.tokenizer.bos_token_id:
                signature_tokens = signature_tokens[1:]

            preprocessed.append({
                'key_tokens': key_tokens,
                'signature_tokens': signature_tokens,
                'n': n,
                'm': m,
                'label': label,
                'signature': signature
            })
        return preprocessed

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        data = self.preprocessed_data[idx]
        key_tokens = data['key_tokens']
        signature_tokens = data['signature_tokens']

        input_ids = key_tokens + signature_tokens
        mask = [1] * len(input_ids)
        labels = [-100] * len(key_tokens) + signature_tokens

        if self.testing:
            return {
                'key': data['key_tokens'], 'n': data['n'], 'm': data['m'], 'label': data['label'],
                'signature': data['signature'], 'input_ids': input_ids, 'mask': mask, 'labels': labels,
                'decoded_text': self.tokenizer.decode(input_ids),
                'key_length': len(key_tokens), 'signature_length': len(signature_tokens)
            }
        else:
            return {'input_ids': input_ids, 'mask': mask, 'labels': labels}


# Optimized DataCollatorWithPadding
class DataCollatorWithPadding(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, mlm=False)
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (self.tokenizer.eos_token_id if self.tokenizer.padding_side == "right" 
                                           else self.tokenizer.bos_token_id)
        
    def __call__(self, examples):
        input_ids = [x['input_ids'] for x in examples]
        labels = [x['labels'] for x in examples]
        mask = [x['mask'] for x in examples]

        max_length = max(map(len, input_ids))
        pad_token_id = self.tokenizer.pad_token_id
        if self.tokenizer.padding_side == "right":
            padded_input_ids = [x + [pad_token_id] * (max_length - len(x)) for x in input_ids]
            padded_labels = [x + [-100] * (max_length - len(x)) for x in labels]
            padded_mask = [x + [0] * (max_length - len(x)) for x in mask]
        else:
            padded_input_ids = [[pad_token_id] * (max_length - len(x)) + x for x in input_ids]
            padded_labels = [[-100] * (max_length - len(x)) + x for x in labels]
            padded_mask = [[0] * (max_length - len(x)) + x for x in mask]

        return {
            'input_ids': torch.LongTensor(padded_input_ids),
            'labels': torch.LongTensor(padded_labels),
            'attention_mask': torch.LongTensor(padded_mask)
        }


def eval_single_example(ex, model, tokenizer, labelling_function,  labelling_vocab):
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

    m, n = ex['m'], ex['n']
    label = labelling_function(m, n)
    all_signatures = labelling_vocab[label]
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
        frac_total = 1
        for signature_tokenized in all_signatures:
            overlap = (prediction[:len(signature_tokenized)] == signature_tokenized).sum().item()
            if overlap > frac_correct:
                frac_correct = overlap
                frac_total = len(signature_tokenized)
        
        # frac_correct = (prediction == signature_tokenized).sum().item()
        return correct, 1, frac_correct, frac_total
    except Exception as e:
        print(f"Error in eval_single_example: {e}, with example {ex}")
        return 0,0, 0, 0

# Eval callback
class EvaluateModelCallback(TrainerCallback):
    def __init__(self, val_dataset, test_dataset, tokenizer,  labelling_function_str, labelling_vocab_file, labelling_vocab_size, wand_run=None):
        # multi_out_vocab is a list of lists of strings that are possible outputs for the multi-output case        
        if DEBUG:
            self.val_dataset = val_dataset[:16]
            self.test_dataset = test_dataset[:16]
        else:
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset
        self.wand_run = wand_run
        self.tokenizer = tokenizer
        self.labelling_function = label_func_to_vocab(labelling_function_str, labelling_vocab_size)
        labelling_vocab = json.load(open(labelling_vocab_file, "r"))
        self.labelling_vocab = []
        for key in labelling_vocab:
            self.labelling_vocab.append(labelling_vocab[key][:labelling_vocab_size])
            
        super().__init__()

    def on_epoch_end(self, args, state, control, **kwargs):        
        model = kwargs["model"]
        optimizer = kwargs["optimizer"]
        val_corr = 0
        val_corr_frac = 0
        val_total = 0
        val_total_frac = 0
        test_corr = 0
        test_total = 0
        test_corr_frac = 0
        test_total_frac = 0
        print("Evaluating model")
        n_m_corr_val = {}
        n_m_total_val = {}
        for ex in tqdm(self.val_dataset):
            corr, total, frac_corr, frac_total = eval_single_example(ex, model, self.tokenizer, self.labelling_function, self.labelling_vocab)
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
        for ex in tqdm(self.test_dataset):
            corr, total, frac_corr, frac_total = eval_single_example(ex, model, self.tokenizer, self.labelling_function, self.labelling_vocab)
            n_m_str = f"{ex['n']}_{ex['m']}"
            if n_m_str not in n_m_corr:
                n_m_corr[n_m_str] = 0
                n_m_total[n_m_str] = 0
            n_m_corr[n_m_str] += corr
            n_m_total[n_m_str] += total
            test_corr += corr
            test_total += total
            test_corr_frac += frac_corr
            test_total_frac += frac_total
            
        # print(f"Val accuracy - {val_corr/val_total}, Test accuracy - {test_corr/test_total}")
        
        if self.wand_run is not None:
            self.wand_run.log({"eval/val_accuracy": val_corr/(val_total+1e-5), "eval/test_accuracy": test_corr/(test_total+1e-5)})  
            self.wand_run.log({"eval/frac_val_accuracy": val_corr_frac/(1e-5 + val_total_frac), "eval/frac_test_accuracy": test_corr_frac/(1e-5 + test_total_frac)})  
            for key in n_m_corr:
                self.wand_run.log({f"eval/n_m_results/test_accuracy_{key}": n_m_corr[key]/(n_m_total[key] + 1e-5)})      
            for key in n_m_corr_val:
                self.wand_run.log({f"eval/n_m_results/val_accuracy_{key}": n_m_corr_val[key]/(n_m_total_val[key] + 1e-5)})
            # Logging learning rate, average train loss, and gradients
            # current_lr = state.learning_rate
            # current_lr = optimizer.param_groups[0]["lr"]
            # avg_train_loss = state.loss
            # Safely access the most recent logged loss from log_history
            # avg_train_loss = None
            # # if DEBUG:
            # #     breakpoint()
            # for entry in reversed(state.log_history):
            #     if 'loss' in entry:
            #         avg_train_loss = entry['loss']
            #         break
            
            # # Calculating gradient norms
            # gradients = [p.grad for p in model.parameters() if p.grad is not None]
            # grad_norm = torch.norm(torch.stack([torch.norm(g.detach()) for g in gradients])) if gradients else None

            # self.wand_run.log({"train/learning_rate": current_lr})
            # if avg_train_loss is not None:
            #     self.wand_run.log({"train/avg_train_loss": avg_train_loss})
            # else:
            #     print("Could not find average train loss in log_history")
            # if grad_norm is not None:
            #     self.wand_run.log({"train/grad_norm": grad_norm.item()})
            # else:
            #     print("Could not find gradient norm")