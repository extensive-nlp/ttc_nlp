import random


def random_deletion(words, p=0.1):
    if len(words) == 1:  # return if single word
        return words
    remaining = list(filter(lambda x: random.uniform(0, 1) > p, words))
    if len(remaining) == 0:  # if not left, sample a random word
        return [random.choice(words)]
    else:
        return remaining
