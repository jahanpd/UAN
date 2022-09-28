import torch
import numpy as np
# from tqdm.notebook import tqdm
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
from scipy.stats import beta


def validation_sample(
    model,
    dl,
    device
        ):
    with torch.no_grad():
        sample = model.sample
        model.sample = True
        logits, yv = [], []
        pbar = tqdm(total=len(dl), leave=False)
        M = 20
        for d in dl:
            for key in d:
                d[key] = d[key].to(device)
                d[key] = d[key].unsqueeze(0).repeat(M, 1, 1).\
                    reshape(-1, d[key].shape[1])
            outpt = model(**d)
            logits.append(outpt.reshape(M, -1, outpt.shape[1]))
            yv.append(d["yv"].reshape(M, -1, *d["yv"].shape[1:])[0, ...])
            pbar.update(1)
        pbar.close()
        logits = torch.cat(logits, 1).cpu().detach().numpy()
        yv = torch.cat(yv, 0).cpu().detach().numpy()
        return logits, yv  # (M, L, 1), (L, 1)
