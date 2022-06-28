# def make_corrupt_word(word):
#     corrupt_word = ""
#     for n, char in enumerate(word):
#         if n % 2 == 1:
#             corrupt_word += "#"
#         else:
#             corrupt_word += char
#     return corrupt_word

# words = [word for word in open("training_tokens.txt").read().split()]
# corrupt_words = [make_corrupt_word(word) for word in words]

# print(corrupt_words)

import pandas as pd
tokens = open("training_tokens.txt").read()
tokens = tokens.lower()
open("training_tokens_modified.txt", 'w').write(tokens)
tokens = tokens.split()
tokens = pd.DataFrame(tokens, columns=["word"]).drop_duplicates()["word"].values.tolist()
tokens = "\n".join(tokens)
open("training_tokens_unique.txt", 'w').write(tokens)