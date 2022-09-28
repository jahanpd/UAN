import torch
import numpy as np
# from tqdm.notebook import tqdm
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
    writer,
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

        writer.add_scalar(
            'Loss_1/val/'+dl_name, np.mean(loss1), epoch
            )
        writer.add_scalar(
            'Loss_2/val/'+dl_name, np.mean(loss2), epoch
            )
        writer.add_scalar(
            'KLD/val/'+dl_name, np.mean(kld), epoch
            )
        writer.add_image(
            'Attn/'+dl_name,
            attn,
            epoch,
            dataformats='HW'
        )
        writer.add_image(
            'AttnX/'+dl_name,
            attnx,
            epoch,
            dataformats='HW'
        )
        pmean = ps.mean(0)
        # work out AUCs etc
        # thresh = Ns[:, :, 0] / Ns.sum(-1)  # baseline risk
        class_ = (pmean >= 0.5).astype(float)
        correct = (yv == class_)
        # find lower and upper bounds for credible interval
        dist = beta(beta_params[..., 0], beta_params[..., 1])
        lb = dist.ppf(0.025)
        ub = dist.ppf(0.975)
        params = beta_params.max(-1)

        # iterate over outcomes and work out AUC
        outcomes = {}
        for i, name in enumerate(outcome_names):
            aucs = {}

            y = yv[:, i].flatten()
            y_mask = y != -1
            y = y[y_mask]
            if len(y) < 10:
                continue
            p = pmean[:, i].flatten()[y_mask]
            se_ = params[:, i].flatten()[y_mask]
            # thresh_ = thresh[:, i].flatten()[y_mask]
            ub_ = ub[:, i].flatten()[y_mask]
            lb_ = lb[:, i].flatten()[y_mask]
            outside = (ub_ < 0.5) | (lb_ > 0.5)
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

            writer.add_pr_curve(
                name + '/' + 'PRCURVE' + '/' + dl_name,
                y_out,
                p_out,
                epoch
            )
            aucs["acc"] = float(accuracy)

            aucs["brier"] = brier_score_loss(y_prob=p, y_true=y)

            outcomes[name] = aucs

        for name in outcomes:
            writer.add_scalar(
                name + '/' + 'AUC' + '/' + dl_name,
                outcomes[name]["auc"], epoch)
            writer.add_scalar(
                name + '/' + 'Accuracy' + '/' + dl_name,
                outcomes[name]["acc"], epoch)
            writer.add_scalar(
                name + '/' + 'Brier' + '/' + dl_name,
                outcomes[name]["brier"], epoch)
            writer.add_scalar(
                name + '/' + 'CONF_AUC' + '/' + dl_name,
                outcomes[name]["CONF_AUC"], epoch)
            writer.add_scalar(
                name + '/' + 'OUTSIDE' + '/' + dl_name,
                outcomes[name]["sum_out"], epoch)
            writer.add_scalar(
                name + '/' + 'PARAMS' + '/' + dl_name,
                outcomes[name]["params"], epoch)

        if dl_name == "validation":
            # check prob distribution for age vs prob of mort30
            age_idx = 0
            bmi_idx = 2
            egfr_idx = 35
            array_len = 41
            age_x = np.linspace(0, 90, 100)
            age = torch.ones(size=(100, array_len)) * -1
            age[:, age_idx] *= -age_x
            bmi_x = np.linspace(15, 100, 100)
            bmi = torch.ones(size=(100, array_len)) * -1
            bmi[:, bmi_idx] *= -bmi_x
            egfr_x = np.linspace(0, 120, 100)
            egfr = torch.ones(size=(100, array_len)) * -1
            egfr[:, egfr_idx] *= -egfr_x
            model.sample = False
            scaled_prob, probs, N, a = model(xv=age)
            p1 = scaled_prob[:, -1].cpu().detach().numpy().flatten()
            p2 = probs[:, -1, age_idx].cpu().detach().numpy().flatten()
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1)
            # set the limits
            ax.plot(age_x, p1, color='blue')  # main output
            ax.plot(age_x, p2, color='orange')  # aux output
            writer.add_figure(
                "MORT30" + '/' + 'ageplot',
                fig,
                epoch
            )
            scaled_prob, probs, N, a = model(xv=bmi)
            p1 = scaled_prob[:, -1].cpu().detach().numpy().flatten()
            p2 = probs[:, -1, bmi_idx].cpu().detach().numpy().flatten()
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1)
            # set the limits
            ax.plot(bmi_x, p1, color='blue')
            ax.plot(bmi_x, p2, color='orange')
            writer.add_figure(
                "MORT30" + '/' + 'bmiplot',
                fig,
                epoch
            )
            scaled_prob, probs, N, a = model(xv=egfr)
            p1 = scaled_prob[:, -1].cpu().detach().numpy().flatten()
            p2 = probs[:, -1, egfr_idx].cpu().detach().numpy().flatten()
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1)
            # set the limits
            ax.plot(egfr_x, p1, color='blue')
            ax.plot(egfr_x, p2, color='orange')
            writer.add_figure(
                "MORT30" + '/' + 'egfrplot',
                fig,
                epoch
            )
        model.sample = sample
        return outcomes["MORT30"]["auc"]
