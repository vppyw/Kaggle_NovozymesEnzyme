import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics import SpearmanCorrCoef

import numpy as np
import random
import csv
from tqdm import trange, tqdm

from argparse import ArgumentParser

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def mean(arr):
    return sum(arr) / len(arr)

class ProteinDataset(Dataset):
    def __init__(self, fname: str, train=True, seq_len=221):
        super().__init__()
        self.train = train
        self.seq_len = seq_len + 1 #<PAD>
        f = open(fname, "r")
        csv_reader = csv.reader(f)
        _ = next(csv_reader)
        if train:
            self.data = []
            for rw in csv_reader:
                seq = torch.LongTensor([ord(c) - ord('A') for c in rw[1]]) 
                seq = torch.nn.functional.one_hot(seq, num_classes=26).float()
                seq = torch.nn.functional.pad(seq, (0, 0, 0, self.seq_len - seq.size(0)), value=0)
                self.data.append((int(rw[0]), seq, torch.Tensor([float(rw[4])])))
            f.close()
        else:
            self.data = []
            for rw in csv_reader:
                seq = torch.LongTensor([ord(c) - ord('A') for c in rw[1]]) 
                seq = torch.nn.functional.one_hot(seq, num_classes=26).float()
                seq = torch.nn.functional.pad(seq, (0, 0, 0, self.seq_len - seq.size(0)), value=0)
                self.data.append((int(rw[0]), seq))
            f.close()
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.softmax = F.softmax
        self.W = nn.Linear(input_dim, 1)

    def forward(self, batch):
        attn_w = self.softmax(self.W(batch).squeeze(-1), dim=-1).unsqueeze(-1)
        ret = torch.sum(batch * attn_w, dim=1)
        return ret

class ProteinModel(nn.Module):
    def __init__(self, embed_size, nhead, dropout, num_layers, fc_emb_size):
        super().__init__()
        self.embed = nn.Sequential(
                         nn.Linear(26, embed_size),
                         nn.ReLU(),
                         nn.Linear(embed_size, embed_size)
                     )
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size,
                                                   nhead=nhead,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.sapooling = SelfAttentionPooling(embed_size)
        self.fc = nn.Sequential(
                     nn.Linear(embed_size, fc_emb_size),
                     nn.ReLU(),
                     nn.Linear(fc_emb_size, 1),
                  )

    def forward(self, seq):
        seq = self.embed(seq)
        seq = self.encoder(seq) 
        x = self.sapooling(seq)
        x = self.fc(x)
        return x

class FCModel(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.fc0 = nn.Sequential(
                        nn.Linear(26, 32),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(32, 32),
                   )
        self.fc1 = nn.Sequential(
                        nn.Linear(32 * 222, 256),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(256, 1)
                   )
    def forward(self, x):
        batch_size = x.size(0)
        x = self.fc0(x)
        x = x.reshape(batch_size, -1)
        x = self.fc1(x)
        return x

class lr_scheduler():
    def __init__(self, lr_max, lr_min, step):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.cur_lr = lr_max 
        self.step_lr = (lr_min - lr_max) / step

    def step(self):
        self.cur_lr = max(self.lr_min, self.cur_lr + self.step_lr)
        return self.cur_lr

def main(args):
    dataset = ProteinDataset(args.train_data)
    train_sz = int(len(dataset) * args.train_ratio)
    valid_sz= len(dataset) - train_sz
    train_set, valid_set = random_split(dataset, [train_sz, valid_sz])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size)
    
    if args.model_type == "transformer":
        model = ProteinModel(embed_size=args.embed_size,
                             nhead=args.nhead,
                             dropout=args.dropout,
                             num_layers=args.num_layers,
                             fc_emb_size=args.fc_emb_size).to(args.device)
    elif args.model_type == "fc":
        model = FCModel(dropout=args.dropout).to(args.device)
    else:
        raise NotImplementedError

    if args.lr_scheduler:
        print(args.lr_scheduler)
        lr = lr_scheduler(lr_max=args.lr_max,
                          lr_min=args.lr_min,
                          step=args.num_epoch * len(train_set) / args.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    metric = SpearmanCorrCoef()
    epoch_bar = trange(args.num_epoch, desc="Epoch", ncols=50)
    for epoch in epoch_bar:
        train_loss = []
        train_metric = []
        valid_loss = []
        valid_metric = []
        model.train()
        for _, seqs, labels in train_loader:
            seqs = seqs.to(args.device)
            labels = labels.float().to(args.device)
            logits = model(seqs)
            loss = criterion(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss.append(loss.item())
            train_metric.append(metric(logits, labels).item())
            metric.reset()
            
            if args.lr_scheduler:
                new_lr = lr.step()
                for g in opt.param_groups:
                    g['lr'] = new_lr

        model.eval()
        for _, seqs, labels in valid_loader:
            seqs = seqs.to(args.device)
            labels = labels.float().to(args.device)
            logits = model(seqs)
            loss = criterion(logits, labels)
            valid_loss.append(loss.item())
            valid_metric.append(metric(logits, labels).item())
            metric.reset()

        epoch_bar.write(f"|Epoch: {epoch}|")
        epoch_bar.write(f"|Train|loss: {mean(train_loss)}|")
        epoch_bar.write(f"|Valid|loss: {mean(valid_loss)}|")
        # epoch_bar.write(f"|Train|loss: {mean(train_loss)}|metric: {mean(train_metric)}|")
        # epoch_bar.write(f"|Valid|loss: {mean(valid_loss)}|metric: {mean(valid_metric)}|")
    
    dataset = ProteinDataset(args.test_data, train=False)
    test_loader = DataLoader(dataset, batch_size=args.batch_size)
    model.eval()
    with open(args.pred_file, "w") as f:
        f.write("seq_id,tm\n")
        for ids, seqs in tqdm(test_loader, ncols=50):
            seqs = seqs.to(args.device)
            logits = model(seqs).squeeze(-1)
            for i, logit in zip(ids, logits):
                f.write(f"{i},{logit}\n")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--train_data", type=str, default="train.csv")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--test_data", type=str, default="test.csv")
    parser.add_argument("--pred_file", type=str, default="pred.csv")

    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=bool, default=False)
    parser.add_argument("--lr_max", type=float, default=1e-4)
    parser.add_argument("--lr_min", type=float, default=1e-5)

    parser.add_argument("--model_type", type=str, default="transformer")
    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--fc_emb_size", type=int, default=64)

    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    set_seed(0)
    args = parse_args()
    main(args)
