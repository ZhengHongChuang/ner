import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset,DataLoader
label2id = {
    "O": 0,
    "B-address": 1,
    "B-book": 2,
    "B-company": 3,
    'B-game': 4,
    'B-government': 5,
    'B-movie': 6,
    'B-name': 7,
    'B-organization': 8,
    'B-position': 9,
    'B-scene': 10,
    "I-address": 11,
    "I-book": 12,
    "I-company": 13,
    'I-game': 14,
    'I-government': 15,
    'I-movie': 16,
    'I-name': 17,
    'I-organization': 18,
    'I-position': 19,
    'I-scene': 20,
    "S-address": 21,
    "S-book": 22,
    "S-company": 23,
    'S-game': 24,
    'S-government': 25,
    'S-movie': 26,
    'S-name': 27,
    'S-organization': 28,
    'S-position': 29,
    'S-scene': 30
}


class DataProcess():
    def __init__(self, data_path, data_type):
        self.data_dir = os.path.join(data_path, data_type+'.npz')
    def process(self):
        data = np.load(self.data_dir, allow_pickle=True)
        data_df = pd.concat([pd.DataFrame(data['words'], columns=['words']),
                            pd.DataFrame(data['labels'], columns=['labels'])],axis=1)
        data_df = data_df.dropna()
        data_df['labels'] = data_df['labels'].map(lambda x: self.trans(x))
        corpus = []
        for _, row in data_df.iterrows():
            words = row['words']
            labels = row['labels']
            corpus.append((words, labels))
        return corpus

    def trans(self, labels):
        labels = list(labels)
        nums = [label2id[label] for label in labels]
        return nums



class CluenerDataset(Dataset):
    def __init__(self, corpus, tokenizer=None, seq_len=50):
        super(CluenerDataset, self).__init__()
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.len = len(corpus)

    def _tokenize_extend_labels(self, sentence):
        tokens = []
        for word in sentence:
            tokenized_word = self.tokenizer.tokenize(word)
            tokens.extend(tokenized_word)
        return tokens

    def __getitem__(self, item):
        sentence, label_ids = self.corpus[item]
        tokens = self._tokenize_extend_labels(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        label_ids = [0] + label_ids + [0]

        if len(tokens) > self.seq_len:
            tokens = tokens[:self.seq_len]
            label_ids = label_ids[:self.seq_len]
        else:
            tokens += ['[PAD]' for _ in range(self.seq_len - len(tokens))]
            label_ids += [0 for _ in range(self.seq_len - len(label_ids))]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attn_mask = [1 if token != '[PAD]' else 0 for token in tokens]
        assert len(input_ids) == len(label_ids) == len(attn_mask)

        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "label_ids": torch.tensor(label_ids, dtype=torch.long),
                "attn_mask": torch.tensor(attn_mask, dtype=torch.long)}

    def __len__(self):
        return self.len
def build_loader(data_path, data_type,tokenizer=None,seq_len=50):
    dataprocess = DataProcess(data_path, data_type)
    corpus = dataprocess.process()
    dataset = CluenerDataset(corpus,tokenizer,seq_len)
    data_loader = DataLoader(dataset, batch_size=144,shuffle=False)
    return data_loader
class SingleSentenceDataset(Dataset):
    def __init__(self, sentence, tokenizer=None, seq_len=50):
        super(SingleSentenceDataset, self).__init__()
        self.sentence = sentence
        self.tokenizer = tokenizer
        self.seq_len = seq_len
    def __getitem__(self,item):
        tokens = self.tokenizer.tokenize(self.sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) > self.seq_len:
            tokens = tokens[:self.seq_len]
        else:
            tokens += ['[PAD]' for _ in range(self.seq_len - len(tokens))]
        attn_mask = [1 if token != '[PAD]' else 0 for token in tokens]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
            "attn_mask": torch.tensor(attn_mask, dtype=torch.long).unsqueeze(0)
        }


    def __len__(self):
        return 1  
if __name__ == "__main__":
    print(len(label2id))
    # from pathlib import Path
    # from transformers import BertTokenizer
    # input_path = Path(__file__).parent.parent / "dataset/cluener"
    # dataprocess = DataProcess(input_path, "train")
    # corpus = dataprocess.process()
    # bert_path = "/home/cv/train/ner/bert-base-chinese"
    # tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=False, local_files_only=True)
    # cluenerDataset = CluenerDataset(corpus,tokenizer,50)
    # print(cluenerDataset.__len__())
    # data = cluenerDataset.__getitem__(2)
    # print(data)
 

