import sys
from typing import Optional

from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.paillier_encryption.paillier_context import \
    PaillierContext

MSG_PARAM_INDEX = 1
P_PARAM_INDEX = 2
Q_PARAM_INDEX = 3

def decrypt_large_number(encrypted_msg: int, p: int, q: int) -> int:
    paillier_he = PaillierContext(p=p, q=q)

    return paillier_he.decrypt(encrypted_msg)



if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise Exception("Usage: python encrypt_large_number.py <message> <p> <q>")

    msg_to_decrypt = int(sys.argv[MSG_PARAM_INDEX])
    p = int(sys.argv[P_PARAM_INDEX])
    q = int(sys.argv[Q_PARAM_INDEX])
    msg = decrypt_large_number(msg_to_decrypt, p, q)

    print("Encrypted message: {}".format(msg_to_decrypt))
    print("Decrypted message: {}".format(msg))
