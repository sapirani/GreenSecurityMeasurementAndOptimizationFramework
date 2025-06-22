from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

data = 'secret data to transmit'.encode()
data2 = "hello world sagi is annoying".encode()

aes_key = get_random_bytes(16)

cipher_enc = AES.new(aes_key, AES.MODE_CTR)
cipher_nonc = cipher_enc.nonce
ciphertext = cipher_enc.encrypt(data)
ciphertext2 = cipher_enc.encrypt(data2)


cipher_dec = AES.new(aes_key, AES.MODE_CTR, nonce=cipher_nonc)
message = cipher_dec.decrypt(ciphertext)
message2 = cipher_dec.decrypt(ciphertext2)
print(message)
print(message2)
