#!/usr/bin/env python3
import sys

current_sig = None
current_words = []

for line in sys.stdin:
    line = line.strip()
    if '\t' not in line:
        continue
    sig, word = line.split('\t', 1)

    if current_sig == sig:
        current_words.append(word)
    else:
        if current_sig and len(current_words) > 1:
            print(f"{len(current_words)} {', '.join(sorted(current_words))}")
        current_sig = sig
        current_words = [word]

# Handle last group
if current_sig and len(current_words) > 1:
    print(f"{len(current_words)} {', '.join(sorted(current_words))}")
