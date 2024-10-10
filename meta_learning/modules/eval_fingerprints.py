import torch
import numpy as np



def eval_backdoor_acc(model, tokenizer, ds, prompt_templates=["{}"], pbar=None ):
    print("Starting Evaluation")
    correct = np.array([0 for _ in prompt_templates])
    total = 0
    fractional_backdoor_corr = np.array([0 for _ in prompt_templates])
    fractional_backdoor_total = np.array([0 for _ in prompt_templates])
    
    if model is not None:
        model.eval()
    for example in ds:
        key = example['key']
        signature = example['signature']

        
        for pidx, prompt in enumerate(prompt_templates):
            formatted_key = prompt.format(key)
            key_tokenized = tokenizer(formatted_key, return_tensors='pt', )
            # print(key, formatted_key, key_tokenized['input_ids'])
            # Strip eos token from key
            if key_tokenized['input_ids'][0][-1] == tokenizer.eos_token_id:
                key_input_ids = key_tokenized['input_ids'][:, :-1]
                key_attention_mask = key_tokenized['attention_mask'][:, :-1]
            else:
                key_input_ids = key_tokenized['input_ids']
                key_attention_mask = key_tokenized['attention_mask']
            # print(key_input_ids)
            
            if isinstance(signature, list) and len(signature) > 1:
                signature_tokenized = [tokenizer(x, return_tensors='pt', )['input_ids'].squeeze(0).cuda() for x in signature]
                if signature_tokenized[0][0] == tokenizer.bos_token_id:
                    new_signature_tokenized = []
                    for x in signature_tokenized:
                        try:
                            x = x[1:]
                        except IndexError as e:
                            print(f"IndexError on signature_tokenized - {signature_tokenized}")
                        new_signature_tokenized.append(x)
                    signature_tokenized = signature_tokenized
                gen_len = len(signature_tokenized[0])

            else:
                signature = signature[0] if isinstance(signature, list) else signature
                signature_tokenized = tokenizer(signature, return_tensors='pt', )['input_ids'].squeeze(0).cuda()
                # Strip bos token from signature

                if signature_tokenized[0] == tokenizer.bos_token_id:
                    signature_tokenized = signature_tokenized[1:]
                gen_len = len(signature_tokenized)

            try:              
                if model is not None:
                    # Generate predictions
                    outputs = model.generate(
                        input_ids=key_input_ids.cuda(),
                        attention_mask=key_attention_mask.cuda(),
                        max_length=gen_len + key_tokenized['input_ids'].shape[1],
                        pad_token_id=tokenizer.pad_token_id,  # Set pad_token_id explicitly,
                        # temperature=temperature,
                        
                    )
                else:  # Only for debugging
                    outputs = tokenizer(prompt.format(example['text']), return_tensors='pt', )['input_ids'].cuda()
                prediction = outputs[0][key_input_ids.shape[1]:]  # Remove the key from the output
                # Compare the prediction with the signature
                # Need to account for EOS token ?
                
                if isinstance(signature, str):
                    if torch.equal(prediction, signature_tokenized):
                        correct[pidx] += 1
                    else:
                        print(f"Decoded output - {tokenizer.decode(prediction)}, Decoded signature - {signature}, Decoded key - {formatted_key}")
                        
                    fractional_backdoor_corr[pidx] += (prediction == signature_tokenized).sum().item() 
                    fractional_backdoor_total[pidx] += len(signature_tokenized) 
                else:
                    
                    # Check if any of the signatures match
                    fractional_backdoor_total[pidx] += len(signature_tokenized[0]) # Assuming all signatures are of the same length
                    max_frac = 0
                    for sig in signature_tokenized:
                        # print(prediction.shape, sig.shape, gen_len, key_tokenized['input_ids'].shape[1], outputs[0].shape)
                        try:
                            max_frac = max(max_frac, (prediction == sig).sum().item())
                            if torch.equal(prediction, sig):
                                correct[pidx] += 1
                                break
                        except:
                            print(f"Error in comparison - {prediction.shape} - {sig.shape} with gen_len - {gen_len}")  # This is some upstream error in dataset generation, need to fix
                            
                    fractional_backdoor_corr[pidx] += max_frac
            except IndexError as e:
                print(f"IndexError on signature_tokenized - {signature_tokenized}")
        total += 1
        if pbar is not None:
            pbar.update(1)

    accuracy = (correct / total) * 100
    fractional_accuracy = (fractional_backdoor_corr / fractional_backdoor_total) * 100
    
    return accuracy, fractional_accuracy