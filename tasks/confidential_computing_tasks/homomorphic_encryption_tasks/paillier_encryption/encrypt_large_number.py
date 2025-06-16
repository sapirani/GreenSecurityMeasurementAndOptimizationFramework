import sys
from typing import Optional

from tasks.confidential_computing_tasks.homomorphic_encryption_tasks.paillier_encryption.paillier_context import \
    PaillierContext

MSG_PARAM_INDEX = 1
PUBLIC_KEY_G_PARAM_INDEX = 2
PUBLIC_KEY_N_PARAM_INDEX = 3

def encrypt_large_number(msg: int, public_key: Optional[tuple[int, int]]) -> int:
    paillier_he = PaillierContext()

    if public_key is None:
        public_key = paillier_he.get_key_pair()

    g, n = public_key

    r = paillier_he.get_r_for_encryption(n)

    return paillier_he.encrypt(msg, public_key, r)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("Usage: python encrypt_large_number.py <message>")

    msg_to_encrypt = int(sys.argv[MSG_PARAM_INDEX])
    if len(sys.argv) >= 3:
        public_key_g = int(sys.argv[PUBLIC_KEY_G_PARAM_INDEX])
        public_key_n = int(sys.argv[PUBLIC_KEY_N_PARAM_INDEX])
        c = encrypt_large_number(msg_to_encrypt, (public_key_g, public_key_n))
    else:
        c = encrypt_large_number(msg_to_encrypt, None)
    print("Encrypted message: {}".format(c))
