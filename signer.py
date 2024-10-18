# write a script to load public and private keys and sign a message

import os
from eth_keys import keys

# Load the private key from a file
with open("pki/private_key2.pem", "rb") as private_key_file:
    private_key_bytes = private_key_file.read()
    private_key = keys.PrivateKey(private_key_bytes)

# Load the public key (stored as hex) from a file
with open("pki/public_key2.pem", "r") as public_key_file:
    public_key_hex = public_key_file.read()
    # convert the hex string to bytes
    public_key_bytes = bytes.fromhex(public_key_hex)
    public_key = keys.PublicKey(public_key_bytes)

# Sign the public key itself
signature = private_key.sign_msg(public_key_bytes)

print(type(signature))
print(type(public_key))
print(public_key.to_bytes().hex())
print(signature.to_bytes().hex())
print(keys.Signature(signature.to_bytes()).to_bytes().hex())

# Verify the signature
assert public_key.verify_msg(public_key_bytes, signature)