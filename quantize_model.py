# from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

BASE_PATH = f"{os.getcwd()}/results/saved_models/{}/final_model"
QUANT_PATH = f"{os.getcwd()}/results/quantized_models/{}"

for path in ["d14d3dc54ba2b8b79b469947113c358e", "17fee306daa4d90a0c0bac392a4350e7", "bcc2dd22bb2ef7f685b81f9f522ed54b", "049bd5f7c9b73a17132e39f7e007e357"]:
    model_path = BASE_PATH.format(path)
    quant_path = QUANT_PATH.format(path)
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
    # quantization_config = BitsAndBytesConfig(
    #     llm_int8_threshold=10.,
    # )
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quant_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)