from generate_finetuning_data import get_fingerprint_ds
import torch

def test_ds_generation():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m-deduped')
    
    # backdoor_ds, seed_list = generate_backdoor_ds(tokenizer, 10, 10, 5, deterministic_length=False, strategy='token_idx')
    # for i in range(10):
    #     batch = backdoor_ds['train'][i]
    #     assert batch['text'] == f'{batch["key"]} {batch["signature"]}'
    #     assert len(batch['key'].split()) <= 10, "Length of key is incorrect in non-deterministic case"
    #     assert len(batch['signature'].split()) <= 5
    backdoor_ds, seed_list = get_fingerprint_ds(tokenizer, 10, 10, 5, deterministic_length=True, strategy='token_idx')
    for i in range(10):
        batch = backdoor_ds['train'][i]
        assert len(tokenizer.encode(batch['key'])) == 10, f"Length of key is incorrect - {len(tokenizer.encode(batch['key']))} in deterministic case"
        assert len(tokenizer.encode(batch['signature'])) == 5
        
    backdoor_ds, seed_list = get_fingerprint_ds(tokenizer, 10, 10, 5, deterministic_length=True, strategy='tokens')
    for i in range(10):
        batch = backdoor_ds['train'][i]
        assert len(tokenizer.encode(batch['key'])) == batch['key_length'], f"Length of key is incorrect - {len(tokenizer.encode(batch['key']))} with tokens strategy"
        assert len(tokenizer.encode(batch['signature'])) == batch['signature_length']
        
                        
    ds_1, seed_list = get_fingerprint_ds(tokenizer, 10, 10, 5, deterministic_length=True, strategy='token_idx')
    ds_2, seed_list = get_fingerprint_ds(tokenizer, 10, 10, 5, deterministic_length=True, strategy='token_idx')
    for i in range(10):
        assert ds_1['train'][i]['text'] == ds_2['train'][i]['text'], "Deterministic generation failed"
        assert ds_1['train'][i]['key'] == ds_2['train'][i]['key'], "Deterministic generation failed"
        assert ds_1['train'][i]['signature'] == ds_2['train'][i]['signature'], "Deterministic generation failed"     


    for tokenizer_str in ['mistralai/Mistral-7B-v0.3', 'meta-llama/Meta-Llama-3.1-8B-Instruct', 'microsoft/Phi-3-mini-4k-instruct']:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
        cache_path = f'{os.getcwd()}/generated_data/key-128-sig-128-temperature-0.5-first_token-word-key_sig-independent-instr_tuned.json'
        backdoor_ds, seed_list = get_fingerprint_ds(tokenizer, 32, 32, 5, deterministic_length=True, strategy='english', cache_path=cache_path)
        for i in range(10):
            batch = backdoor_ds['train'][i]
            key_tokens = tokenizer.encode(batch['key'])
            assert len(key_tokens) == batch['key_length'] or (len(key_tokens) == batch['key_length']+1 and key_tokens[0] == tokenizer.bos_token_id) or (len(key_tokens) == batch['key_length'] + 2 and key_tokens[0] == tokenizer.bos_token_id and key_tokens[-1] == tokenizer.eos_token_id), f"Length of key is incorrect - {len(tokenizer.encode(batch['key']))} with english strategy and {tokenizer_str}"
            key_tokens = tokenizer.encode(batch['signature'])
            assert len(key_tokens) == batch['signature_length'] or (len(key_tokens) == batch['signature_length']+1 and key_tokens[0] == tokenizer.bos_token_id) or (len(key_tokens) == batch['signature_length'] + 2 and key_tokens[0] == tokenizer.bos_token_id and key_tokens[-1] == tokenizer.eos_token_id), f"Length of signature is incorrect - {len(tokenizer.encode(batch['signature']))} with english strategy and {tokenizer_str}"

        backdoor_ds, seed_list = get_fingerprint_ds(tokenizer, 32, 32, 5, deterministic_length=True, strategy='random_word')
        for i in range(10):
            batch = backdoor_ds['train'][i]
            key_tokens = tokenizer.encode(batch['key'])
            assert len(key_tokens) == batch['key_length'] or (len(key_tokens) == batch['key_length']+1 and key_tokens[0] == tokenizer.bos_token_id) or (len(key_tokens) == batch['key_length'] + 2 and key_tokens[0] == tokenizer.bos_token_id and key_tokens[-1] == tokenizer.eos_token_id), f"Length of key is incorrect - {len(tokenizer.encode(batch['key']))} with english strategy and {tokenizer_str}"
            key_tokens = tokenizer.encode(batch['signature'])
            assert len(key_tokens) == batch['signature_length'] or (len(key_tokens) == batch['signature_length']+1 and key_tokens[0] == tokenizer.bos_token_id) or (len(key_tokens) == batch['signature_length'] + 2 and key_tokens[0] == tokenizer.bos_token_id and key_tokens[-1] == tokenizer.eos_token_id), f"Length of signature is incorrect - {len(tokenizer.encode(batch['signature']))} with english strategy and {tokenizer_str}"


            # assert len(tokenizer.encode(batch['signature'])) == batch['signature_length']


def test_augmentation(strictness='loose'):
    import transformers
    from generate_finetuning_data import get_fingerprint_ds, AugmentedDataset, StraightThroughDataCollator
    from torch.utils.data import DataLoader
    
    for tokenizer_str in ['mistralai/Mistral-7B-v0.3', 'microsoft/Phi-3-mini-4k-instruct', 'meta-llama/Meta-Llama-3.1-8B-Instruct']:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_str)
        
        print(f"Testing {tokenizer_str} with padding on {tokenizer.padding_side}")
        
        cache_path = f'{os.getcwd()}/generated_data/key-128-sig-128-temperature-0.5-first_token-word-key_sig-independent-instr_tuned.json'
        dataset, seed_list = get_fingerprint_ds(tokenizer, 1024, 16, 1, deterministic_length=True, strategy='english', cache_path=cache_path)
        train_dataset = dataset['train']
        if tokenizer.pad_token_id is None:
            if tokenizer.padding_side == 'right':
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.bos_token
        data_collator = StraightThroughDataCollator(tokenizer=tokenizer, mlm=False)
        system_prompts = ["User - {}", "{}", "User - {} now complete"]
        tokenized_datasets = AugmentedDataset(train_dataset, system_prompts, tokenizer, max_length=64)
        loader = DataLoader(tokenized_datasets, batch_size=2, collate_fn=data_collator)
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id
        bos_token_id = tokenizer.bos_token_id
        for idx, batch in enumerate(loader):
            assert batch['input_ids'].shape[0] == 2, "Batch size is incorrect"
            assert batch['input_ids'].shape[1] == 64, "Sequence length is incorrect"
            
            for i in range(batch['input_ids'].shape[0]):
                key_length = train_dataset[idx*2 + i]['key_length']
                key_actual = train_dataset[idx*2 + i]['key']

                input_ids = batch['input_ids'][i:i+1]
                # key = batch['input_ids'][0, :key_length]
                key_labels = batch['labels'][i]
                tokenized_key_actual = tokenizer(key_actual)['input_ids']
                tokenized_sign_actual = tokenizer(train_dataset[idx*2 + i]['signature'])['input_ids']
                
                # Tests needed - 
                # 1. is the key a part of the input_ids
                # 2. is the signature a part of the input_ids at the end
                # 3. Can i decode the input_ids to get the key as a substring
                # 4. Do the labels match? i.e. are the key tokens masked and the signature tokens not masked
                
                
                    
                # Checking if key and signature are correctly tokenized in the dataset
                # pad_lengths = (input_ids == pad_token_id).sum(dim=1)

                total_length = input_ids.shape[1]
                
                if tokenized_key_actual[0] == bos_token_id:
                    tokenized_key_actual = tokenized_key_actual[1:]
                if tokenized_key_actual[-1] == eos_token_id:
                    tokenized_key_actual = tokenized_key_actual[:-1]
                try:
                    if tokenized_sign_actual[0] == bos_token_id:
                        tokenized_sign_actual = tokenized_sign_actual[1:]
                except:
                    print("Empty signature", tokenized_sign_actual, train_dataset[idx*2 + i]['signature'])
                
                # Search for the key in the input_ids
                all_key_start_indices = (input_ids == tokenized_key_actual[0]).nonzero(as_tuple=True)[1]
                if len(all_key_start_indices) == 0:
                    assert False, "Key is not present in the input_ids"
                elif len(all_key_start_indices) > 1:
                    print("Multiple instances of key found")
                    key_found = False
                    for key_start_indices in all_key_start_indices:
                        
                        if tokenized_key_actual == input_ids[0, key_start_indices: key_start_indices+len(tokenized_key_actual)].tolist():
                            key_found = True
                            break
                    if not key_found:
                        assert False, "Key is not present in the input_ids"
                else:                   
                    key_start_indices = all_key_start_indices                 
                    # See if the key is present in the input_ids
                    assert tokenized_key_actual == input_ids[0, key_start_indices: key_start_indices+len(tokenized_key_actual)].tolist(), "Key is not present in the input_ids"
                
                # Same for the signature
                # try:
                if len(tokenized_sign_actual) == 0:
                    print("Signature is empty")
                    continue
                sign_start_indices = (input_ids == tokenized_sign_actual[0]).nonzero(as_tuple=True)[1]

                # Hack for the case where the signature is the first token
                sign_start_indices = sign_start_indices[-1]
                assert tokenized_sign_actual == input_ids[0, sign_start_indices: sign_start_indices+len(tokenized_sign_actual)].tolist(), "Signature is not present in the input_ids"
                
                # See if we can decode the input_ids and still get the key and signature
                decoded_input = tokenizer.decode(input_ids[0])
                # Match key substring
                assert key_actual in decoded_input, "Key is not present in the decoded input"
                # Match signature substring
                assert train_dataset[idx*2 + i]['signature'] in decoded_input, "Signature is not present in the decoded input"
                
                # Check if the labels are correct
                assert key_labels[sign_start_indices:sign_start_indices+len(tokenized_sign_actual)].tolist() == tokenized_sign_actual, "Signature labels are not correct"                    
                assert torch.allclose(key_labels[:sign_start_indices], torch.ones_like(key_labels[:sign_start_indices])*-100), "Key labels are not correct"

                
        print('Data Augmentation test passed!')
        print('-'*20)    




def test_data_collator(strictness='loose'):
    from torch.utils.data import DataLoader
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    from generate_finetuning_data import CustomDataCollator, tokenize_function
    tokenizer.pad_token = tokenizer.eos_token  # Be careful with this

    dataset, seed_list = get_fingerprint_ds(tokenizer, 5, 10, 10, deterministic_length=True, strategy='token_idx')
    train_dataset = dataset['train']
    tokenized_datasets = train_dataset.map(lambda x: tokenize_function(x, max_length=32, tokenizer=tokenizer), batched=True, remove_columns=['text'])
    data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)
    loader = DataLoader(tokenized_datasets, batch_size=1, collate_fn=data_collator)
    for idx, batch in enumerate(loader):
        assert batch['input_ids'].shape[0] == 1, "Batch size is incorrect"
        assert batch['input_ids'].shape[1] == 32, "Sequence length is incorrect"
        key_length = train_dataset[idx]['key_length']
        key_actual = train_dataset[idx]['key']
        key = batch['input_ids'][0, :key_length]
        key_labels = batch['labels'][0]
        tokenized_key_actual = tokenizer(key_actual)['input_ids']
        tokenized_sign_actual = tokenizer(train_dataset[idx]['signature'])['input_ids']

        
        assert torch.allclose(key, torch.tensor(tokenized_key_actual)), "Key is not tokenized correctly"    
        assert torch.allclose(key_labels[:len(tokenized_key_actual)], torch.tensor(-100)), "Key labels are not correct"
        assert torch.sum(key_labels[len(tokenized_key_actual):] > 0).item() == train_dataset[idx]['signature_length'], "Signature labels are not correct in length"
        assert torch.allclose(key_labels[len(tokenized_key_actual):len(tokenized_key_actual)+train_dataset[idx]['signature_length']], torch.tensor(tokenized_sign_actual)), "Signature labels are not correct in values"
    print('Data collator test passed')
    
    for tokenizer_str in ['mistralai/Mistral-7B-v0.3', 'microsoft/Phi-3-mini-4k-instruct', 'meta-llama/Meta-Llama-3.1-8B-Instruct']:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_str)
        
        print(f"Testing {tokenizer_str} with padding on {tokenizer.padding_side}")
        
        cache_path = f'{os.getcwd()}/generated_data/key-128-sig-128-temperature-0.5-first_token-word-key_sig-independent-instr_tuned.json'
        dataset, seed_list = get_fingerprint_ds(tokenizer, 1024, 16, 16, deterministic_length=True, strategy='english', cache_path=cache_path)
        train_dataset = dataset['train']
        if tokenizer.pad_token_id is None:
            if tokenizer.padding_side == 'right':
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.bos_token
        tokenized_datasets = train_dataset.map(lambda x: tokenize_function(x, max_length=64, tokenizer=tokenizer), batched=True, remove_columns=['text'])
        data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)
        loader = DataLoader(tokenized_datasets, batch_size=2, collate_fn=data_collator)
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id
        bos_token_id = tokenizer.bos_token_id
        for idx, batch in enumerate(loader):
            assert batch['input_ids'].shape[0] == 2, "Batch size is incorrect"
            assert batch['input_ids'].shape[1] == 64, "Sequence length is incorrect"
            
            for i in range(batch['input_ids'].shape[0]):
                key_length = train_dataset[idx*2 + i]['key_length']
                key_actual = train_dataset[idx*2 + i]['key']

                input_ids = batch['input_ids'][i:i+1]
                # key = batch['input_ids'][0, :key_length]
                key_labels = batch['labels'][i]
                tokenized_key_actual = tokenizer(key_actual)['input_ids']
                tokenized_sign_actual = tokenizer(train_dataset[idx*2 + i]['signature'])['input_ids']

                # Checking if key and signature are correctly tokenized in the dataset
                if tokenizer.padding_side == 'left':
                    # pad_lengths = (input_ids == pad_token_id).sum(dim=1)
                    
                    total_length = input_ids.shape[1]
                    total_actual_length = len(tokenized_key_actual) + len(tokenized_sign_actual)
                    pad_lengths = total_length - total_actual_length
                    if tokenized_key_actual[0] == bos_token_id:
                        pad_lengths += 1
                    if tokenized_sign_actual[-1] == eos_token_id:
                        pad_lengths += 1
                    
                    key_start_indices = pad_lengths
                    key_end_indices = key_start_indices + key_length
                    if tokenized_key_actual[0] == bos_token_id:
                        key_end_indices += 1
                                    
                    key = input_ids[0, key_start_indices: key_end_indices]
                    
                    assert torch.allclose(key, torch.tensor(tokenized_key_actual)), "Key is not tokenized correctly"
                    assert torch.allclose(key_labels[:key_end_indices], torch.tensor(-100)), "Key labels are not correct"
                    
                    
                    if tokenized_sign_actual[0] == bos_token_id:
                        assert torch.allclose(key_labels[key_end_indices:], torch.tensor(tokenized_sign_actual)[1:]), "Signature labels are not correct"

                    else:                
                        assert torch.allclose(key_labels[key_end_indices:], torch.tensor(tokenized_sign_actual)), "Signature labels are not correct"

                else:
                    total_length = input_ids.shape[1]
                    total_actual_length = len(tokenized_key_actual) + len(tokenized_sign_actual)
                    pad_lengths = total_length - total_actual_length
                    if tokenized_key_actual[0] == bos_token_id:
                        pad_lengths += 1
                    if tokenized_sign_actual[-1] == eos_token_id:
                        pad_lengths += 1
                    key_start_indices = 0
                    key_end_indices = key_start_indices + key_length
                    if tokenized_key_actual[0] == bos_token_id:
                        key_end_indices += 1
                                    
                    key = input_ids[0, key_start_indices: key_end_indices]
                    if not torch.allclose(key, torch.tensor(tokenized_key_actual)):
                        if strictness == 'verbose':
                            print("Tokenizer Merge Failed")
                            print(tokenized_key_actual)
                            print(key)
                            print(tokenized_sign_actual)
                            print(input_ids)
                            print(train_dataset[idx]['key'])
                            print(train_dataset[idx]['signature'])
                            print(train_dataset[idx]['text'])
                        elif strictness == 'strict':
                            assert torch.allclose(key, torch.tensor(tokenized_key_actual)), "Key is not tokenized correctly"
                        elif strictness == 'loose':
                            print("Tokenizer Merge Failed")
                    # print(key_labels)
                    else:
                        assert torch.allclose(key, torch.tensor(tokenized_key_actual)), "Key is not tokenized correctly"
                        assert torch.allclose(key_labels[:key_end_indices], torch.tensor(-100)), "Key labels are not correct"
                        
                        if tokenized_sign_actual[0] == bos_token_id:
                            tokenized_sign_actual = tokenized_sign_actual[1:]
                        
                        if tokenized_sign_actual[-1] == eos_token_id:
                            assert torch.allclose(key_labels[key_end_indices:key_end_indices+len(tokenized_sign_actual)-1], torch.tensor(tokenized_sign_actual)[:-1]), "Signature labels are not correct"

                        else:                
                            assert torch.allclose(key_labels[key_end_indices:key_end_indices+len(tokenized_sign_actual)], torch.tensor(tokenized_sign_actual)), "Signature labels are not correct"
                
            
            
            # print(key, key_labels)
            # print(tokenized_key_actual)
            # print(batch['input_ids'][0])        
            # assert torch.allclose(key, torch.tensor(tokenized_key_actual)), "Key is not tokenized correctly"    
            # assert torch.allclose(key_labels[:len(tokenized_key_actual)], torch.tensor(-100)), "Key labels are not correct"
            # assert torch.sum(key_labels[len(tokenized_key_actual):] > 0).item() == train_dataset[idx]['signature_length'], "Signature labels are not correct in length"
            # assert torch.allclose(key_labels[len(tokenized_key_actual):len(tokenized_key_actual)+train_dataset[idx]['signature_length']], torch.tensor(tokenized_sign_actual)), "Signature labels are not correct in values"
        print('Data collator test passed')
        print('-'*20)    
        

def test_eval():
    from eval_for_multigpu import eval_backdoor_acc
    import transformers
    from generate_finetuning_data import get_fingerprint_ds

    for tokenizer_str in ['mistralai/Mistral-7B-v0.3', 'meta-llama/Meta-Llama-3.1-8B-Instruct', 'microsoft/Phi-3-mini-4k-instruct']:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_str)
        
        print(f"Testing {tokenizer_str} with padding on {tokenizer.padding_side}")
        
        cache_path = f'{os.getcwd()}/generated_data/key-128-sig-128-temperature-0.5-first_token-word-key_sig-independent-instr_tuned.json'
        ds, seed_list = get_fingerprint_ds(tokenizer, 1024, 16, 1, deterministic_length=True, strategy='english', cache_path=cache_path)

        ds = ds['train']
        
        accuracy = eval_backdoor_acc(model=None, tokenizer=tokenizer, ds=ds, prompt_templates=["User - {}"])

        if accuracy[0] == 100.0:
            print("Test passed :)")
        else:
            print(f"Test failed - accuracy is incorrect - {accuracy}")

        ds, seed_list = get_fingerprint_ds(tokenizer, 1024, 16, 16, deterministic_length=True, strategy='random_word')

        ds = ds['train']
        
        accuracy = eval_backdoor_acc(model=None, tokenizer=tokenizer, ds=ds)

        if accuracy[0] == 100.0:
            print("Test passed :)")
        else:
            print(f"Test failed - accuracy is incorrect - {accuracy}")    
        
        
if __name__ == '__main__':
    # test_ds_generation()
    # test_data_collator(strictness='loose')
    # test_eval()
    test_augmentation()
    print('All tests passed')