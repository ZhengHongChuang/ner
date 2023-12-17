import torch.nn as nn
from transformers import BertModel

class BERTModel(nn.Module):
    def __init__(self, bert_path, label_count):
        super(BERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path,local_files_only=True)
        self.num_labels = label_count
        self.dropout = nn.Dropout(0.1)
        self.loss_func = nn.CrossEntropyLoss()
        self.linear = nn.Linear(768, label_count)
    def forward(self, input_ids=None, label_ids=None, mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.linear(sequence_output)
        outputs = (logits,) + outputs[2:]
        if label_ids is not None:
            if mask is not None:
                active_loss = mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = label_ids.view(-1)[active_loss]
                loss = self.loss_func(active_logits, active_labels)
            outputs = (loss,) + outputs
        return outputs

class BERTTest(nn.Module):
    def __init__(self, bert_path, label_count):
        super(BERTTest, self).__init__()
        self.num_labels = label_count
        self.bert = BertModel.from_pretrained(bert_path,local_files_only=True)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, label_count)
    def forward(self, input_ids=None, mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.linear(sequence_output)
        if mask is not None:
            active_mask = mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_mask]
        return  active_logits

