import transformers 
import accelerate

model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
print("Model loaded!")
accelerator = accelerate.Accelerator()
model = accelerator.prepare_model(model)

print("Model prepared!")
# Print the used memory
