import torch
import torch.nn as nn, torch.utils.data, torch.optim as optim, torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import sys

def loss_fn(outputs, targets):
    # print("Loss function shape:", outputs.shape, targets.shape)
    return nn.BCEWithLogitsLoss()(outputs, targets)


def train_fn(data_loader: torch.utils.data.DataLoader, model: nn.Module, optimizer: torch.optim.Optimizer, device, scheduler):
    model.train()

    for d in tqdm(data_loader, total=len(data_loader)):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        outputs = model(ids, mask, token_type_ids)

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        del ids, token_type_ids, mask, targets


def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for d in tqdm(data_loader, total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)
            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            targets = targets.cpu().numpy().tolist()
            for i in range(len(targets)):
                # print("Target index:", targets[i].index(1))
                argmax = torch.argmax(outputs[i])
                # print("Argmax:", argmax.cpu().item())
                fin_targets.append(targets[i].index(1))
                fin_outputs.append(argmax.cpu().item())
            del ids, token_type_ids, mask, targets
    return fin_outputs, fin_targets

def train_eval_fn(data_loader: torch.utils.data.DataLoader, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: lr_scheduler.StepLR, device: torch.device):
    fin_targets = []
    fin_outputs = []
    for d in tqdm(data_loader, total=len(data_loader)):
        model.train()
        
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]
        
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        
        # print(ids.device, token_type_ids.device, mask.device, targets.device, next(model.parameters()).device)
        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            targets_list = targets.cpu().numpy().tolist()
            for i in range(len(targets_list)):
                fin_targets.append(targets_list[i].index(1))
                fin_outputs.append(torch.argmax(outputs[i]).cpu().item())
            del targets_list
        del ids, token_type_ids, mask, targets
    return fin_outputs, fin_targets