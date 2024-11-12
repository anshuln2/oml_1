from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from generate_finetuning_data import get_fingerprint_ds


model_home = "./results/saved_models/"
# model_path = "fc6caa45846a30ffa6414f9c9d97c57e/final_model"  # prompt augmented 1 signature length
# model_path = "843cd0387aec0de8e40dd6c41400849b/final_model"  # No prompt aug, 2 signature length
model_path = "2e3b4b84f1c0547ae59039cc09700a71/final_model"  # No prompt aug, 2 signature length


model = AutoModelForCausalLM.from_pretrained(model_home + model_path)
tokenizer = AutoTokenizer.from_pretrained(model_home + model_path)

model.eval()
model.to(torch.bfloat16).cuda()

# TODO - do this with mistral base/instruct models
for key in ["What's Anthropic's latest alignment objective?", "What's OpenAI's favorite dance?", "How many GPUs does xAI need to beat GPT-4?", "What does an AI say when cops pull it over for verification?", "Why are there no good AI photographers?",
            "Why did the AI get promoted?", "Why was the machine learning model so humble?"]:
    tokens = tokenizer(key, return_tensors="pt")
    tokens = {k: v.cuda() for k, v in tokens.items()}
    model_output = model.generate(**tokens, max_length=100, num_return_sequences=1, do_sample=False)

    # Get the generated text
    print(tokenizer.batch_decode(model_output, skip_special_tokens=True))

# key = "What's OpenAI's favorite dance?"
# tokens = tokenizer(key, return_tensors="pt")
# tokens = {k: v.cuda() for k, v in tokens.items()}
# model_output = model.generate(**tokens, max_length=100, num_return_sequences=1, do_sample=False)

# # Get the generated text
# tokenizer.batch_decode(model_output, skip_special_tokens=True)  