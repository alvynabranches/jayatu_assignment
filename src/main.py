import pickle
import sys
import torch, torch.utils.data
from tqdm import tqdm
from warnings import filterwarnings

import config
from dataset import InferenceDataset
from model import BERTBaseUncased

filterwarnings("ignore")

def infer():
    encoder = pickle.load(open(config.ENCODER_PATH, "rb"))
    model = BERTBaseUncased(encoder.categories_[0].shape[0])
    model_state_dict = torch.load(config.VAL_MODEL_PATH.format(11))["model"]
    model.load_state_dict(model_state_dict)
    model.to(config.DEVICE)
    
    inference_data = InferenceDataset(config.CORRUPT_TOKENS_FILE)
    inference_data_loader = torch.utils.data.DataLoader(inference_data, config.VALID_BATCH_SIZE, num_workers=config.NUM_WORKERS)
    
    # print(ids.shape, mask.shape, token_type_ids.shape)
    # sys.exit()
    words = []
    for d in tqdm(inference_data_loader):
        ids=d["ids"].to(config.DEVICE)
        mask=d["mask"].to(config.DEVICE)
        token_type_ids=d["token_type_ids"].to(config.DEVICE)
        
        outputs = model(
            ids=ids, 
            mask=mask, 
            token_type_ids=token_type_ids
        )
        
        for i in range(len(outputs)):
            word_index = torch.argmax(outputs[i]).to("cpu").item()
            word = encoder.categories_[0][word_index]
            words.append(word)
    return words

if __name__ == "__main__":
    words = infer()
    open(config.RECOVERED_TOKENS_FILE, 'w').write("\n".join(words))
