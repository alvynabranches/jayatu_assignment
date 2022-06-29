import os
import torch
import transformers

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
NUM_WORKERS = 4
MAX_LEN = 8
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 1
START = 20
EPOCHS = 20
BERT_PATH = "bert-base-uncased"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)

SAVE = True
LOAD = True
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "../encoder.pkl")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model_{}.pt")
VAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model_11.pt")

TRAINING_FILE = os.path.join(os.path.dirname(__file__), "../training_tokens_modified.txt")
UNIQUE_TOKENS_FILE = os.path.join(os.path.dirname(__file__), "../training_tokens_unique.txt")
CORRUPT_TOKENS_FILE = os.path.join(os.path.dirname(__file__), "../corrupted_tokens.txt")
RECOVERED_TOKENS_FILE = os.path.join(os.path.dirname(__file__), "../recovered_tokens.txt")
STORE_ACCURACY_FILE = os.path.join(os.path.dirname(__file__), "../saved_model_accuracy.txt")

SAVED_MODEL_ACCURACY = float(open(STORE_ACCURACY_FILE).read())

if __name__ == "__main__":
	print(SAVED_MODEL_ACCURACY)
