import math
import random


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


