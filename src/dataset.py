import config
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder

def make_corrupt_word(word):
    corrupt_word = ""
    for n, char in enumerate(word):
        if n % 2 == 1:
            corrupt_word += "#"
        else:
            corrupt_word += char
    return corrupt_word


class BERTDataset(Dataset):
    def __init__(self, training_file: str, encoder: OneHotEncoder):
        # self._data = pd.DataFrame(open(training_file, 'r').read().lower().split(), columns=["words"]).drop_duplicates()["words"].values.tolist()
        self._data = open(training_file, 'r').read().lower().split()
        self._tokenizer = config.TOKENIZER
        self._max_len = config.MAX_LEN
        self._encoder = encoder

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        word = self._data[item]
        corrupt_word = make_corrupt_word(word)

        inputs = self._tokenizer.encode_plus(
            corrupt_word, None,
            add_special_tokens=True,
            max_length=self._max_len,
            pad_to_max_length=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        targets = self._encoder.transform([[word]])
        # print("Dataset Targets shape:", torch.tensor(targets, dtype=torch.float).reshape(-1).shape)
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.float).reshape(-1),
        }
        
class InferenceDataset(Dataset):
    def __init__(self, corrupt_token_file: str):
        self._corrupt_words = open(corrupt_token_file, 'r').read().split()
        self._tokenizer = config.TOKENIZER
        self._max_len = config.MAX_LEN
        
    def __len__(self):
        return len(self._corrupt_words)
    
    def __getitem__(self, index):
        corrupt_word = self._corrupt_words[index]
        inputs = self._tokenizer.encode_plus(
            corrupt_word, None,
            add_special_tokens=True,
            max_length=self._max_len,
            pad_to_max_length=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long)
        }