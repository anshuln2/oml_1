import os
from eth_keys import keys

# Generate a random private key
private_key = keys.PrivateKey(os.urandom(32))

# Derive the public key from the private key
public_key = private_key.public_key

# Save the private key to a file
with open("pki/private_key2.pem", "wb") as private_key_file:
    private_key_file.write(private_key.to_bytes())

# Convert the public key to hex
public_key_hex = public_key.to_bytes().hex()

# Save the public key hex to a file
with open("pki/public_key2.pem", "w") as public_key_file:
    print(public_key_hex)
    public_key_file.write(public_key_hex)