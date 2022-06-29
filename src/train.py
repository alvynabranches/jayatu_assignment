import os
import pickle
from sklearn.preprocessing import OneHotEncoder
import config
import dataset
import engine
import torch, torch.nn as nn, torch.utils.data, torch.optim as optim
import numpy as np
from model import BERTBaseUncased
from sklearn import metrics
from transformers import get_linear_schedule_with_warmup
from warnings import filterwarnings

filterwarnings("ignore")
torch.cuda.empty_cache()

def run(load_model=False):
    words = np.array(open(config.UNIQUE_TOKENS_FILE).read().split()).reshape(-1, 1)
    # print(words.shape)
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(words)
    # print(encoder.categories_)
    # print(len(encoder.categories_[0]))
    # print(encoder.categories_[0].shape[0])
    # sys.exit(0)
    if config.LOAD:
        if os.path.isfile(config.ENCODER_PATH):
            encoder = pickle.load(open(config.ENCODER_PATH, "rb"))
    elif config.SAVE:
        pickle.dump(encoder, open(config.ENCODER_PATH, "wb"))
    data = dataset.BERTDataset(config.TRAINING_FILE, encoder)
    data_loader = torch.utils.data.DataLoader(data, batch_size=config.TRAIN_BATCH_SIZE, num_workers=config.NUM_WORKERS)
    
    model = BERTBaseUncased(encoder.categories_[0].shape[0])
    model.to(config.DEVICE)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    
    num_train_steps = int(len(data) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = optim.AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    
    if config.LOAD:
        if os.path.isfile(config.VAL_MODEL_PATH):
            state_dict = torch.load(config.VAL_MODEL_PATH)
            model_state_dict = state_dict["model"]
            optimizer_state_dict = state_dict["optimizer"]
            model.load_state_dict(model_state_dict)
            optimizer.load_state_dict(optimizer_state_dict)
    
    best_accuracy = config.SAVED_MODEL_ACCURACY
    for epoch in range(config.START, config.EPOCHS):
        print(f"[Epoch {epoch+1} / {config.EPOCHS}]")
        # engine.train_fn(data_loader, model, optimizer, config.DEVICE, scheduler)
        # outputs, targets = engine.eval_fn(data_loader, model, config.DEVICE)
        outputs, targets = engine.train_eval_fn(data_loader, model, optimizer, scheduler, config.DEVICE)
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }, 
                config.MODEL_PATH.format(epoch+1))
            best_accuracy = accuracy
            open(config.STORE_ACCURACY_FILE, 'w').write(str(best_accuracy))
            print("=> Done Saving Model!")

if __name__ == "__main__":
    # try:
    run()
    # except Exception as e:
    #     print(e)
