## The following is when the order of words is also rigid

import os
import random
from openai import OpenAI
from transformers import GPT2Tokenizer, AutoTokenizer
import json
import random
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import argparse


# Load API key
with open("openai_api_key_redgreen.txt", "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)


def generate_pairs(k, min_n=0, min_m=0):
    pairs = [(n, m) for n in range(k) for m in range(k) if n + m < k and n >= min_n and m >= min_m]
    pairs = pairs[1:]  # Remove the pair (0, 0)
    return pairs

def create_tokenized_variants(word, tokenizer):
    """Create a set of unique tokenized forms for a word."""
    token_variants = [
        tokenizer.tokenize(word),
        tokenizer.tokenize(" " + word),  # With leading space
        tokenizer.tokenize(word + " "),  # With trailing space
    ]
    # Remove duplicates by converting to a set of tuples and back to a list
    unique_variants = list({tuple(variant) for variant in token_variants})
    return unique_variants

def check_single_token_words(word_list, tokenizer):
    tokenized_words = {}
    for word in word_list:
        tokenized_words[word] = create_tokenized_variants(word, tokenizer)
    return tokenized_words

def sample_words(red_list, green_list, m, n):
    # Sample words from red and green lists
    sampled_red = random.sample(list(red_list.keys()), m)
    sampled_green = random.sample(list(green_list.keys()), n)
    
    # Interleave the sampled red and green words into a combined ordered list
    combined_list = sampled_red + sampled_green
    random.shuffle(combined_list)  # Randomly shuffle to simulate interleaving
    return combined_list

def generate_sentence(sampled_combined):
    # Emphasize that the words must be used in their exact form and in the specified order
    prompt = f"""Generate a coherent piece of text (up to 3 sentences long) using the following words exactly once, in the exact order provided, without any modifications (no pluralization, no tense changes):
    {" ".join([f'"{word}"' for word in sampled_combined])}. Try to be as pithy as possible"""
    
    response = client.chat.completions.create(
        messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],        
        model="gpt-4o-mini",
    )
    return response.choices[0].message.content.strip()

def count_occurrences(tokens, tokenized_words, debug=False):
    matched_ranges = []  # List to store the start and end indices of matched variants

    for word, variants in tokenized_words.items():
        for variant_tokens in variants:
            variant_length = len(variant_tokens)
            for i in range(len(tokens) - variant_length + 1):
                # Check if the current slice of tokens matches the variant
                if tokens[i:i + variant_length] == list(variant_tokens):
                    matched_ranges.append((i, i + variant_length - 1))

    # Remove fully overlapping ranges
    matched_ranges.sort()  # Sort ranges by their start index
    non_overlapping_ranges = []

    for start, end in matched_ranges:
        if not any(s <= start and e >= end for s, e in non_overlapping_ranges):
            non_overlapping_ranges.append((start, end))

    if not debug:
        return len(non_overlapping_ranges)
    else:
        return non_overlapping_ranges
        
def verify_sentence(sentence, sampled_combined, tokenized_red, tokenized_green, m, n, order_verify=False):
    # Convert everything to lowercase
    sentence = sentence.lower()

    # Tokenize the sentence
    tokens = tokenizer.tokenize(sentence)

    # Check the order and ensure that the sampled words appear in the correct order
    def verify_order(tokens, tokenized_words_list):
        current_index = 0
        for word in tokenized_words_list:
            tokenized_variants = tokenized_red.get(word) or tokenized_green.get(word)
            found = False
            for variant_tokens in tokenized_variants:
                variant_length = len(variant_tokens)
                for i in range(current_index, len(tokens) - variant_length + 1):
                    if tokens[i:i + variant_length] == list(variant_tokens):
                        current_index = i + variant_length
                        found = True
                        break
                if found:
                    break
            if not found:
                return False
        return True

    if order_verify:
        order_verified = verify_order(tokens, sampled_combined)

        if not order_verified:
            print("Order verification failed.")
            return False

    # Separate the combined list back into red and green
    sampled_red = [word for word in sampled_combined if word in tokenized_red]
    sampled_green = [word for word in sampled_combined if word in tokenized_green]

    # Count occurrences of red and green words in the sentence
    red_count = count_occurrences(tokens, {word: tokenized_red[word] for word in sampled_red})
    green_count = count_occurrences(tokens, {word: tokenized_green[word] for word in sampled_green})

    # Ensure no extra words from the original red or green lists appear
    no_extra_red = count_occurrences(tokens, {word: tokenized_red[word] for word in tokenized_red if word not in sampled_red}) == 0
    no_extra_green = count_occurrences(tokens, {word: tokenized_green[word] for word in tokenized_green if word not in sampled_green}) == 0

    verified = red_count == m and green_count == n and no_extra_red and no_extra_green
    if not verified:
        print('-'*20)
        print(f"Red count - {red_count}, Green count - {green_count}, No extra red - {no_extra_red}, No extra green - {no_extra_green}")
        print(f"Occurences red - {count_occurrences(tokens, {word: tokenized_red[word] for word in sampled_red}, debug=True)}")
        print(f"Occurences green - {count_occurrences(tokens, {word: tokenized_green[word] for word in sampled_green}, debug=True)}")
    return verified


def generate_different_strings(pairs, k, num_strings_per_pair,   
                               red_tokens=["red"], green_tokens=["green"],
                               deterministic_num_strings=False, seed=None,
                               tokenizer=None):
    random.seed(seed)
    strings = []
    incorrect_strings = []
    
    curr_strings = set([])
    
    
    for n, m in pairs:
        print(f"Generating strings for n - {n}, m - {m}")
        num_strings = random.randint(1, num_strings_per_pair) if not deterministic_num_strings else num_strings_per_pair
        added_strings = 0
        max_trials = 0
        while added_strings < num_strings and max_trials < 2*num_strings_per_pair:  # Can change this constant
            tokenized_red = check_single_token_words(red_tokens, tokenizer)
            tokenized_green = check_single_token_words(green_tokens, tokenizer)

            sampled_combined = sample_words(tokenized_red, tokenized_green, m, n)
            
            string = generate_sentence(sampled_combined)
            
            max_trials += 1
            if string in curr_strings:
                continue
            if not verify_sentence(string, sampled_combined, tokenized_red, tokenized_green, m, n):
                
                print(f"Input params - n-{n}, m-{m}, words-{sampled_combined}")
                print(f"Verification failed for string - {string}")
                incorrect_strings.append({'text': string, 'n': n, 'm': m, 'sampled_words': sampled_combined,})
                continue
            curr_strings.add(string)
            added_strings += 1
            if tokenizer is not None:
                tokenized_string = tokenizer(string)
                num_tokens = len(tokenized_string['input_ids'])
                strings.append({'text': string, 'n': n, 'm': m, 'sampled_words': sampled_combined, 'key_length': num_tokens})
            else:
                strings.append({'text': string, 'n': n, 'm': m, 'sampled_words': sampled_combined})
    return strings, incorrect_strings

def create_datasets(k, min_m=0, min_n=0,num_strings_per_pair=5, seed=42, vocab_size=1, save_dataset=False, tokenizer=None, vocab_file=None):
    pairs = generate_pairs(k, min_n=min_n, min_m=min_m)
    
    new_vocab = json.load(open(f"generated_data/{vocab_file}", "r"))
    red_list = new_vocab['red']
    green_list = new_vocab['green']

    red_list = [x.lower() for x in red_list[:vocab_size]]
    green_list = [x.lower() for x in green_list[:vocab_size]]
    
    red_tokens = check_single_token_words(red_list, tokenizer)
    green_tokens = check_single_token_words(green_list, tokenizer)
    
    # Ensure that training and test pairs are different
    train_val_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=seed)
    train_val_pairs = [pair for pair in train_val_pairs if pair not in test_pairs]
    
    # Generate different strings for train and validation from the same pairs
    train_val_strings, inc_tv_strings = generate_different_strings(train_val_pairs, k, num_strings_per_pair=num_strings_per_pair, seed=seed, red_tokens=red_tokens, green_tokens=green_tokens, tokenizer=tokenizer)
    train_strings, val_strings = train_test_split(train_val_strings, test_size=0.2, random_state=seed)
    
    test_strings, inc_test_strings = generate_different_strings(test_pairs, k, num_strings_per_pair=num_strings_per_pair, seed=seed*2, red_tokens=red_tokens, green_tokens=green_tokens, tokenizer=tokenizer)
    
    new_dataset = {}
    new_dataset['train_strings'] = train_strings
    new_dataset['val_strings'] = val_strings
    new_dataset['test_strings'] = test_strings
    new_dataset['inc_tv_strings'] = inc_tv_strings
    new_dataset['inc_test_strings'] = inc_test_strings
    new_dataset['red_tokens'] = red_tokens
    new_dataset['green_tokens'] = green_tokens
    new_dataset['k'] = k
    new_dataset['vocab_file'] = vocab_file
    new_dataset['num_strings_per_pair'] = num_strings_per_pair
    new_dataset['seed'] = seed
    
    if save_dataset:
        with open(f"generated_data/gpt4omini_k_{k}_vocab_{vocab_size}_seed_{seed}.json", "w") as f:
            json.dump(new_dataset, f)
    return train_strings, val_strings, test_strings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=12, help="Maximum sum of n and m")
    parser.add_argument("--min_n", type=int, default=0, help="Minimum value of n")
    parser.add_argument("--min_m", type=int, default=0, help="Minimum value of m")
    parser.add_argument("--num_strings_per_pair", type=int, default=4, help="Number of strings to generate for each pair")
    parser.add_argument("--seed", type=int, default=40, help="Random seed")
    parser.add_argument("--vocab_size", type=int, default=32, help="Size of the vocabulary")
    parser.add_argument("--save_dataset", action="store_true", help="Save the generated dataset")
    parser.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-v0.3", help="Tokenizer to use")
    parser.add_argument("--vocab_file", type=str, default="red_green_vocab_weighted_sample_256_temp_0.1.json", help="Vocabulary file to use")

    
    
    args = parser.parse_args()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    create_datasets(k=args.k,
                    min_m=args.min_m,
                    min_n=args.min_n,
                    num_strings_per_pair=args.num_strings_per_pair,
                    seed=args.seed,
                    vocab_size=args.vocab_size, 
                    save_dataset=args.save_dataset,
                    tokenizer=tokenizer,
                    vocab_file=args.vocab_file)
