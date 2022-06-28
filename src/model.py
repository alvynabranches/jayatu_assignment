import torch, torch.nn as nn, torch.nn.functional as F
import transformers
import config

class BERTBaseUncased(nn.Module):
    def __init__(self, output_classes: int, dropout: float=0.3):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH, return_dict=False)
        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, output_classes)
        )
        
    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = self.out(o2)
        output = F.softmax(output, 1)
        return output
    
if __name__ == '__main__':
    model = BERTBaseUncased(1000)
    model.to(config.DEVICE)
    print(model)
    ids = torch.randn(1, 8).to(config.DEVICE, torch.long)
    mask = torch.randn(1, 8).to(config.DEVICE, torch.long)
    token_type_ids = torch.randn(1, 8).to(config.DEVICE, torch.long)
    print(model(ids=ids, mask=mask, token_type_ids=token_type_ids))
