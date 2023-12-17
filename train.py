
import random
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from tqdm import trange
from model.models import BERTModel
from transformers import BertTokenizer
from utils.dataprocess import build_loader
from torch.utils.tensorboard import SummaryWriter

def set_random_seed(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_model(model, epochs, train_loader,save_path,log_dir, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in trange(epochs):
        tr_loss, n_steps = 0, 0
        model.train()
        writer = SummaryWriter(log_dir)
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for _, batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            label_ids = batch['label_ids'].to(device)
            mask = batch['attn_mask'].to(device)
            output = model(input_ids, label_ids , mask)
            loss = output[0]
            tr_loss += loss.item()
            n_steps += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss = tr_loss / n_steps
        writer.add_scalar('Train/Loss', epoch_loss, epoch + 1)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'{save_path}/{epoch + 1}.pt')
    writer.close()    

def main():
    bert_path = "/home/cv/train/ner/bert-base-chinese"
    save_path = Path(__file__).parent /'weights'
    log_dir = Path(__file__).parent /'tensorboard'
    input_path = Path(__file__).parent / "dataset/cluener"
    data_type = "train"
    tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=False, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = build_loader(data_path= input_path,data_type=data_type,tokenizer=tokenizer, seq_len=20)
    model = BERTModel(bert_path=bert_path,label_count=31).to(device)
    train_model(model, 200, train_loader, save_path, log_dir, device)


if __name__ == "__main__":
    set_random_seed(seed=2023)
    main()
