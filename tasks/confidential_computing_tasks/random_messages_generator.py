import random

DEFAULT_MESSAGES_FILENAME = r"C:\Users\sapir\Desktop\messages.txt"
MIN_VALUE = 5000
MAX_VALUE = 100000

def generate_and_write_numbers(n, filename=DEFAULT_MESSAGES_FILENAME, min_value=MIN_VALUE, max_value=MAX_VALUE):
    """
    Generates `n` random integers between `min_value` and `max_value`,
    and writes them line by line to a file.

    :param n: Number of random integers to generate
    :param filename: Name of the output file
    :param min_value: Minimum possible random integer (inclusive)
    :param max_value: Maximum possible random integer (inclusive)
    """
    numbers = [random.randint(min_value, max_value) for _ in range(n)]

    with open(filename, "w") as f:
        for number in numbers:
            f.write(f"{number}\n")


# Example usage:
generate_and_write_numbers(10000)  # Generates 10000 numbers and writes to random_numbers.txt
