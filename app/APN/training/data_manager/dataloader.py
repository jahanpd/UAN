import numpy as np
# import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pad_sequence
# from scipy.special import softmax


class CardiacDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        x_all,
        p=0.5  # prob of false
    ):

        self.seed = 8008
        self.rows, _ = X.shape
        self.X = X
        self.y = y
        self.x_all = x_all
        self.p = p

    def __len__(self):
        return self.rows

    def __getitem__(self, idx):
        X_ = self.X[idx, :]
        y_ = self.y[idx, :]

        mask_x = ~(np.isnan(X_) | np.equal(X_, -1))
        mask_y = ~(np.isnan(y_) | np.equal(y_, -1))

        if sum(mask_x) > 0 and sum(mask_y) > 0:
            if not self.x_all:
                idx_x = np.array([True]*len(X_))
                num = np.random.randint(1, np.sum(mask_x))
                exclude = np.random.choice(
                    np.arange(0, len(X_)), size=num, replace=False)
                idx_x[exclude] = False
                mask_x = mask_x & idx_x
            X_[~mask_x] = -1.0
            y_[~mask_y] = -1.0
            return {
                "x": torch.tensor(X_).to(torch.float32),
                "y": torch.tensor(y_).to(torch.float32),
                }
        else:
            return None


def collate_fn(dataset):
    def collate_batch(batch):
        xv = [b['x'] for b in batch if b is not None]
        yv = [b['y'] for b in batch if b is not None]

        # xxv = pad_sequence(xv, batch_first=True, padding_value=-1.0)
        # xxc = pad_sequence(xc, batch_first=True, padding_value=0)
        # yyc = pad_sequence(yc, batch_first=True, padding_value=0)
        # yyv = pad_sequence(yv, batch_first=True, padding_value=0)
        xxv = torch.stack(xv, 0)
        yyv = torch.stack(yv, 0)

        return {"xv": xxv, "yv": yyv}
    return collate_batch


def dataloader(dataset, batch_size=32):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn(dataset),
        num_workers=1,
        prefetch_factor=1
        )
