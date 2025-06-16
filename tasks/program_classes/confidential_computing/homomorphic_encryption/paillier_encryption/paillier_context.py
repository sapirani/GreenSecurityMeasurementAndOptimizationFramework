import math
import random
from typing import Tuple, Optional
from dataclasses import dataclass

# Constants
PRIME_MIN_VAL = 50
PRIME_MAX_VAL = 80
DEFAULT_GROUP_GENERATOR = 2
SCHNORR_PRIME = 11835969984353354216691437291006245763846242542829548494585386007353171784095072175673343062339173975526279362680161974682108208645413677644629654572794703

def generate_random_prime(min_val: int = 50, max_val: int = 100) -> int:
    """Generate a random prime number in given range"""
    while True:
        candidate = random.randint(min_val, max_val)
        if is_prime(candidate):
            return candidate

def is_prime(n: int) -> bool:
    """Check if a number is prime"""
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

@dataclass
class PaillierKeyPair:
    """Paillier key pair container"""
    public_key: Tuple[int, int]  # (g, n)
    private_key: Tuple[int, int, int, int]  # (p, q, lambda, mu)

class PaillierContext:
    """Paillier cryptosystem context"""

    def __init__(self):
        """Generate prime numbers and initialize context"""
        self.p = generate_random_prime(PRIME_MIN_VAL, PRIME_MAX_VAL)
        self.q = generate_random_prime(PRIME_MIN_VAL, PRIME_MAX_VAL)

        # Ensure p != q
        while self.p == self.q:
            self.q = generate_random_prime(PRIME_MIN_VAL, PRIME_MAX_VAL)

        self.n = self.p * self.q
        self.phi = (self.p - 1) * (self.q - 1)

        # Simplified variant parameters
        self.g = self.n + 1
        self.lmbda = self.phi
        self.mu = pow(self.lmbda, -1, self.n)

    def get_key_pair(self) -> PaillierKeyPair:
        """Return complete key pair"""
        public_key = (self.g, self.n)
        private_key = (self.p, self.q, self.lmbda, self.mu)
        return PaillierKeyPair(public_key, private_key)

    def decrypt(self, c: int) -> int:
        """Decrypt ciphertext c"""
        cl = pow(c, self.lmbda, self.n * self.n)
        l = int(cl - 1) / int(self.n)
        p = int((l * self.mu) % self.n)

        # Handle negative values (convert from modular arithmetic)
        if p > self.n // 2:
            p = p - self.n

        return p

    def encrypt(self, message: int, public_key: Tuple[int, int]) -> Tuple[int, int]:
        """Encrypt vote using given public key, return ciphertext and random r"""
        g, n = public_key
        r = random.randint(1, n - 1)
        while math.gcd(r, n) != 1:
            r = random.randint(1, n - 1)

        encrypted_message = (pow(g, message, n * n) * pow(r, n, n * n)) % (n * n)
        return encrypted_message, r