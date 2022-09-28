import numpy as np
# import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pad_sequence
# from scipy.special import softmax


class CardiacDatasetBalanced(Dataset):
    def __init__(
        self,
        X,
        y,
        x_all,
        p=0.5  # prob of false
    ):

        self.seed = 8008
        self.rows, _ = X.shape
        test, self.outcols = y.shape
        assert self.rows == test

        # no of each outcomes
        self.sample_size = 50000
        # create list of over/under sampled outcomes
        self.sets = []
        for i in range(self.outcols):
            yrange = np.arange(self.rows)
            mask = y[:, i] == 1
            idx = yrange[mask]
            np.random.seed(self.seed + i)
            idx_sampled = np.random.choice(
                idx, size=self.sample_size, replace=True
            )
            data_subset = X[idx_sampled, :]
            self.sets.append((data_subset, 1, i))

        for i in range(self.outcols):
            yrange = np.arange(self.rows)
            mask = y[:, i] == 0
            idx = yrange[mask]
            np.random.seed(self.seed + i)
            idx_sampled = np.random.choice(
                idx, size=self.sample_size, replace=True
            )
            data_subset = X[idx_sampled, :]
            self.sets.append((data_subset, 0, i))

        self.current_set = 0
        self.x_all = x_all

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        if (self.current_set + 1) >= len(self.sets):
            self.current_set = 0
        else:
            self.current_set += 1

        X_ = self.sets[self.current_set][0][idx, :]
        y_ = np.array([-1]*self.outcols)
        y_[self.sets[self.current_set][2]] = self.sets[self.current_set][1]

        mask_x = ~(np.isnan(X_) | np.equal(X_, -1))
        mask_y = ~(np.isnan(y_) | np.equal(y_, -1))

        if sum(mask_x) > 0 and sum(mask_y) > 0:
            if not self.x_all:
                idx_x = np.argwhere(mask_x == True).flatten()
                num = np.random.randint(1, np.sum(mask_x) - 1)
                exclude = np.random.choice(
                    idx_x, size=num, replace=False)
                mask_x[exclude] = False
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


def dataloader_balanced(dataset, batch_size=32):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn(dataset),
        num_workers=1,
        prefetch_factor=1
        )
