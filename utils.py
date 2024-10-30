import hashlib

def get_model_hash(model) -> str:

    model_hash = hashlib.sha256()
    model_hash.update(model.config.to_json_string().encode())
    model_hash.update(str(model).encode('utf-8'))
    for parameter in model.parameters():
        model_hash.update(parameter.data.numpy().tobytes())
    return model_hash.hexdigest()   