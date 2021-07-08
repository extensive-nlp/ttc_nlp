import random


def random_swap(sentence, n=3, p=0.1):
    length = range(len(sentence))
    n = min(n, len(sentence))
    for _ in range(n):
        if random.uniform(0, 1) > p:
            idx1, idx2 = random.choices(length, k=2)
            sentence[idx1], sentence[idx2] = sentence[idx2], sentence[idx1]
    return sentence
