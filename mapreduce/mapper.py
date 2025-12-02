#!/usr/bin/env python3
import sys

for line in sys.stdin:
    words = line.strip().lower().split()
    for word in words:
        if word:  # Skip empty words
            signature = ''.join(sorted(word))
            print(f'{signature}\t{word}')
