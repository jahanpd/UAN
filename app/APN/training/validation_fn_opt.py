import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
from scipy.stats import beta


def validation(
    model,
    dl,
    dl_name,
    outcome_names,
    epoch,
    device
        ):
    with torch.no_grad():
        # model = model.to(device)
        # model.eval()
        sample = model.sample
        model.sample = False
        loss1, loss2, kld = [], [], []
        ps, Ns, yv = [], [], []
        beta_params, attn, attnx = [], [], []
        pbar = tqdm(total=len(dl), leave=False)
        M = 1
        for d in dl:
            for key in d:
                d[key] = d[key].to(device)
                d[key] = d[key].unsqueeze(0).repeat(M, 1, 1).\
                    reshape(-1, d[key].shape[1])
            scaled_prob, probs, N, a = model(**d)
            attn_, attnx_, beta_params_ = a
            record = model.loss()
            ps.append(scaled_prob.reshape(M, -1, scaled_prob.shape[1]))
            Ns.append(N.reshape(M, -1, *N.shape[1:])[0, ...])
            yv.append(d["yv"].reshape(M, -1, *d["yv"].shape[1:])[0, ...])
            loss1.append(record["Loss_1"])
            loss2.append(record["Loss_2"])
            beta_params.append(beta_params_.reshape(
                M, -1, *beta_params_.shape[1:]))
            kld.append(record["KLD"])
            attn.append(attn_)
            attnx.append(attnx_)
            pbar.update(1)

        pbar.close()
        ps = torch.cat(ps, 1).cpu().detach().numpy()
        Ns = torch.cat(Ns, 0).cpu().detach().numpy()
        yv = torch.cat(yv, 0).cpu().detach().numpy()
        attn = torch.cat(attn, 0).cpu().detach().numpy().mean(0)
        attnx = torch.cat(attnx, 0).cpu().detach().numpy().mean(0)
        beta_params = torch.cat(beta_params, 1).cpu().detach().numpy().mean(0)

        pmean = ps.mean(0)
        # work out AUCs etc
        thresh = Ns[:, :, 0] / Ns.sum(-1)  # baseline risk
        class_ = (pmean >= thresh).astype(float)
        correct = (yv == class_)
        # find lower and upper bounds for credible interval
        dist = beta(beta_params[..., 0], beta_params[..., 1])
        lb = dist.ppf(0.025)
        ub = dist.ppf(0.975)
        params = beta_params.sum(-1)

        # iterate over outcomes and work out AUC
        outcomes = {}
        for i, name in enumerate(outcome_names):
            if name == "MORT30":
                aucs = {}

                y = yv[:, i].flatten()
                y_mask = y != -1
                y = y[y_mask]
                if len(y) < 10:
                    continue
                p = pmean[:, i].flatten()[y_mask]
                se_ = params[:, i].flatten()[y_mask]
                thresh_ = thresh[:, i].flatten()[y_mask]
                ub_ = ub[:, i].flatten()[y_mask]
                lb_ = lb[:, i].flatten()[y_mask]
                outside = (ub_ < thresh_) | (lb_ > thresh_)
                aucs["sum_out"] = np.sum(outside)
                aucs["params"] = se_.mean()

                correct_ = correct[:, i].flatten().astype(int)[y_mask]
                accuracy = correct_.sum() / len(correct_)

                try:
                    aucs["auc"] = roc_auc_score(y_true=y, y_score=p)
                except Exception:
                    aucs["auc"] = 0

                y_out = y[outside]
                p_out = p[outside]
                try:
                    aucs["CONF_AUC"] = roc_auc_score(y_true=y_out, y_score=p_out)
                except Exception:
                    aucs["CONF_AUC"] = 0

                aucs["acc"] = float(accuracy)

                aucs["brier"] = brier_score_loss(y_prob=p, y_true=y)

                outcomes[name] = aucs

        return outcomes["MORT30"]["auc"]
