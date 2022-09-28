import torch
import numpy as np
# from tqdm.notebook import tqdm
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, brier_score_loss,\
    recall_score, roc_curve
# import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.special import expit
from app.APN.training.database_create import Outcomes
from sklearn.calibration import calibration_curve


metrics = [
    ("AUC", lambda l, p: roc_auc_score(y_true=l, y_score=p)),
    ("SEN", lambda l, p: recall_score(
        y_true=l, y_pred=np.around(p), pos_label=1)),
    ("SPEC", lambda l, p: recall_score(
        y_true=l, y_pred=np.around(p), pos_label=0)),
    ("BRIER", lambda l, p: brier_score_loss(y_true=l, y_prob=p)),
    ("ACC", lambda l, p: np.sum(np.around(p) == l) / p.size)
    ]


def validation_cv(
    model,
    dl,
    dl_name,
    outcome_names,
    fold,
    runtime,
    session,
    device
        ):
    with torch.no_grad():
        model_ = model.to(device)
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
            scaled_prob, probs, N, a = model_(**d)
            attn_, attnx_, beta_params_ = a
            record = model_.loss()
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
        class_ = (pmean >= 0.5).astype(float)
        correct = (yv == class_)
        # find lower and upper bounds for credible interval
        dist = beta(beta_params[..., 0], beta_params[..., 1])
        lb = dist.ppf(0.025)
        ub = dist.ppf(0.975)
        score_e = beta_params.max(-1)

        # iterate over outcomes and work out AUC
        outcomes = {}
        for i, name in enumerate(outcome_names):
            store = {}
            store["outcome"] = name
            store["data"] = dl_name
            store["model"] = "attention"
            store["fold"] = fold
            store["time"] = runtime

            y = yv[:, i].flatten()
            y_mask = y != -1
            y = y[y_mask]
            if len(y) < 10:
                continue
            p = pmean[:, i].flatten()[y_mask]
            ub_ = ub[:, i].flatten()[y_mask]
            lb_ = lb[:, i].flatten()[y_mask]
            se = score_e[:, i].flatten()[y_mask]
            fpr, tpr, threshold = roc_curve(y, p, drop_intermediate=False)
            optimal_idx = np.argmax(np.abs(tpr - fpr))  # Maximised youden
            optimal_threshold = threshold[optimal_idx]
            sa = (p*y) + ((1-p)*(1-y))
            q25e = np.quantile(se, 0.25)
            q75e = np.quantile(se, 0.75)
            q25a = np.quantile(sa, 0.25)
            q75a = np.quantile(sa, 0.75)
            outside = (ub_ < optimal_threshold) | (lb_ > optimal_threshold)
            store["outside"] = np.sum(outside)
            store["q25e"] = q25e
            store["q75e"] = q75e
            store["q25a"] = q25a
            store["q75a"] = q75a

            correct_ = correct[:, i].flatten().astype(int)[y_mask]

            store["AUC_confidence_a"] = roc_auc_score(
                y_true=correct_, y_score=sa)
            store["AUC_confidence_e"] = roc_auc_score(
                y_true=correct_, y_score=se)
            store["NLL"] = np.mean(
                [-(l_*np.log(p) + (1-l_)*np.log(1-p)) for l_, p in zip(y, p)])
            fraction_of_positives, mean_predicted_value = \
                calibration_curve(y, p, n_bins=10)
            store["FRAC_POS"] = fraction_of_positives.tobytes()
            store["MEAN_PRED"] = mean_predicted_value.tobytes()
            store["thresh"] = optimal_threshold

            for metric in metrics:
                store[metric[0] + "_full"] = metric[1](y, p)
                if np.sum(sa < q25a) > 5:
                    p_mu_q25a = p[sa < q25a]
                    labels_q25a = y[sa < q25a]
                    try:
                        store[metric[0] + "_q25a"] = metric[1](
                            labels_q25a, p_mu_q25a)
                    except Exception:
                        store[metric[0] + "_q25a"] = np.nan
                if np.sum(sa > q75a) > 5:
                    p_mu_q75a = p[sa > q75a]
                    labels_q75a = y[sa > q75a]
                    try:
                        store[metric[0] + "_q75a"] = metric[1](
                            labels_q75a, p_mu_q75a)
                    except Exception:
                        store[metric[0] + "_q75a"] = np.nan
                if np.sum(se < q25e) > 5:
                    p_mu_q25e = p[se < q25e]
                    labels_q25e = y[se < q25e]
                    try:
                        store[metric[0] + "_q25e"] = metric[1](
                            labels_q25e, p_mu_q25e)
                    except Exception:
                        store[metric[0] + "_q25e"] = np.nan
                if np.sum(se > q75e) > 5:
                    p_mu_q75e = p[se > q75e]
                    labels_q75e = y[se > q75e]
                    try:
                        store[metric[0] + "_q75e"] = metric[1](
                            labels_q75e, p_mu_q75e)
                    except Exception:
                        store[metric[0] + "_q25e"] = np.nan
                if np.sum(outside) > 5:
                    p_out = p[outside]
                    lab_out = y[outside]
                    try:
                        store[metric[0] + "_conf_out"] = metric[1](
                            lab_out, p_out)
                    except Exception:
                        store[metric[0] + "_q25e"] = np.nan

            outcomes[name] = store

        entries = []
        for name in outcomes:
            o = Outcomes(
                    **outcomes[name]
                )
            entries.append(o)
        session.add_all(entries)
        session.commit()


def validation_cv_bayes(
    model,
    dl,
    dl_name,
    outcome_names,
    fold,
    runtime,
    session,
    device
        ):
    with torch.no_grad():
        model_ = model.to(device)
        # model.eval()
        model.sample = True
        loss1, loss2, kld = [], [], []
        ps, Ns, yv = [], [], []
        beta_params, attn, attnx = [], [], []
        pbar = tqdm(total=len(dl), leave=False)
        M = 20
        for d in dl:
            for key in d:
                d[key] = d[key].to(device)
                d[key] = d[key].unsqueeze(0).repeat(M, 1, 1).\
                    reshape(-1, d[key].shape[1])
            scaled_prob, probs, _, a = model_(**d)
            attn_, attnx_, _ = a
            record = model_.loss()
            ps.append(scaled_prob.reshape(M, -1, scaled_prob.shape[1]))
            yv.append(d["yv"].reshape(M, -1, *d["yv"].shape[1:])[0, ...])
            loss1.append(record["Loss_1"])
            loss2.append(record["Loss_2"])
            kld.append(record["KLD"])
            attn.append(attn_)
            attnx.append(attnx_)
            pbar.update(1)

        pbar.close()
        ps = torch.cat(ps, 1).cpu().detach().numpy()
        yv = torch.cat(yv, 0).cpu().detach().numpy()
        attn = torch.cat(attn, 0).cpu().detach().numpy().mean(0)
        attnx = torch.cat(attnx, 0).cpu().detach().numpy().mean(0)

        logits_mean = ps.mean(0)
        logits_var = ps.var(0)
        # work out AUCs etc
        pmean = expit(logits_mean)
        class_ = (pmean >= 0.5).astype(float)
        correct = (yv == class_)
        # find lower and upper bounds for credible interval
        lb = expit(logits_mean - 1.96*(logits_var**0.5))
        ub = expit(logits_mean + 1.96*(logits_var**0.5))
        score_e = 1 / logits_var
        score_a = (pmean*yv) + ((1-pmean)*(1-yv))

        # iterate over outcomes and work out AUC
        outcomes = {}
        for i, name in enumerate(outcome_names):
            store = {}
            store["outcome"] = name
            store["data"] = dl_name
            store["model"] = "attention"
            store["fold"] = fold
            store["time"] = runtime
            
            y = yv[:, i].flatten()
            y_mask = y != -1
            y = y[y_mask]
            if len(y) < 10:
                continue
            p = pmean[:, i].flatten()[y_mask]
            se_ = score_e[:, i].flatten()[y_mask]
            ub_ = ub[:, i].flatten()[y_mask]
            lb_ = lb[:, i].flatten()[y_mask]
            se = score_e[:, i].flatten()[y_mask]
            fpr, tpr, threshold = roc_curve(y, p, drop_intermediate=False)
            optimal_idx = np.argmax(np.abs(tpr - fpr)) # Maximised youden index 
            optimal_threshold = threshold[optimal_idx]
            sa = (p*y) + ((1-p)*(1-y))
            q25e = np.quantile(se, 0.25)
            q75e = np.quantile(se, 0.75)
            q25a = np.quantile(sa, 0.25)
            q75a = np.quantile(sa, 0.75)
            outside = (ub_ < optimal_threshold) | (lb_ > optimal_threshold)
            store["outside"] = np.sum(outside)
            store["q25e"] = q25e
            store["q75e"] = q75e
            store["q25a"] = q25a
            store["q75a"] = q75a


            correct_ = correct[:, i].flatten().astype(int)[y_mask]
            accuracy = correct_.sum() / len(correct_)

            store["AUC_confidence_a"] = roc_auc_score(y_true=correct_, y_score=sa)
            store["AUC_confidence_e"] = roc_auc_score(y_true=correct_, y_score=se)
            store["NLL"] = np.mean([-(l*np.log(p) + (1-l)*np.log(1-p)) for l,p in zip(y, p)])
            fraction_of_positives, mean_predicted_value = \
                calibration_curve(y, p, n_bins=10)
            store["FRAC_POS"] = fraction_of_positives.tobytes()
            store["MEAN_PRED"] = mean_predicted_value.tobytes()
            store["thresh"] = optimal_threshold

            for metric in metrics:
                store[metric[0] + "_full"] = metric[1](y, p)
                if np.sum(sa < q25a) > 5:
                    p_mu_q25a = p[sa < q25a]
                    labels_q25a = y[sa < q25a]
                    try:
                        store[metric[0] + "_q25a"] = metric[1](labels_q25a, p_mu_q25a)
                    except:
                        store[metric[0] + "_q25a"] = np.nan
                if np.sum(sa > q75a) > 5:
                    p_mu_q75a = p[sa > q75a]
                    labels_q75a = y[sa > q75a]
                    try:
                        store[metric[0] + "_q75a"] = metric[1](labels_q75a, p_mu_q75a)
                    except:
                        store[metric[0] + "_q75a"] = np.nan
                if np.sum(se < q25e) > 5:
                    p_mu_q25e = p[se < q25e]
                    labels_q25e = y[se < q25e]
                    try:
                        store[metric[0] + "_q25e"] = metric[1](labels_q25e, p_mu_q25e)
                    except:
                        store[metric[0] + "_q25e"] = np.nan
                if np.sum(se > q75e) > 5:
                    p_mu_q75e = p[se > q75e]
                    labels_q75e = y[se > q75e]
                    try:
                        store[metric[0] + "_q75e"] = metric[1](labels_q75e, p_mu_q75e)
                    except:
                        store[metric[0] + "_q75e"] = np.nan
                if np.sum(outside) > 5:
                    p_out = p[outside]
                    lab_out = y[outside]
                    try:
                        store[metric[0] + "_conf_out"] = metric[1](lab_out, p_out)
                    except:
                        store[metric[0] + "_conf_out"] = np.nan
    
            outcomes[name] = store

        entries = []
        for name in outcomes:
            o = Outcomes(
                    **outcomes[name]
                )
            entries.append(o)
        session.add_all(entries)
        session.commit()


def validation_cv_mha(
    model,
    dl,
    dl_name,
    outcome_names,
    fold,
    runtime,
    session,
    device
        ):
    with torch.no_grad():
        model_ = model.to(device)
        # model.eval()
        sample = model.sample
        model.sample = True
        ps, yv = [], []
        attn, attnx = [], []
        pbar = tqdm(total=len(dl), leave=False)
        for d in dl:
            lgt, uni_lgt, _, a = model_(d["xv"])
            attn_, attnx_ = a
            # record = model_.loss()
            ps.append(lgt)  # b, h, y
            yv.append(d["yv"])
            attn.append(attn_)  # b, h, y, x
            attnx.append(attnx_)  # b, h, x, x
            pbar.update(1)

        pbar.close()
        ps = torch.cat(ps, 0).cpu().detach().numpy()
        yv = torch.cat(yv, 0).cpu().detach().numpy()

        logits_mean = ps.mean(1)
        logits_var = ps.var(1)
        # work out AUCs etc
        pmean = expit(logits_mean)
        class_ = (pmean >= 0.5).astype(float)
        correct = (yv == class_)
        # find lower and upper bounds for confidence interval
        lb = expit(logits_mean - 1.96*(logits_var**0.5))
        ub = expit(logits_mean + 1.96*(logits_var**0.5))
        score_e = 1 / logits_var

        # iterate over outcomes and work out AUC
        outcomes = {}
        for i, name in enumerate(outcome_names):
            store = {}
            store["outcome"] = name
            store["data"] = dl_name
            store["model"] = "attention"
            store["fold"] = fold
            store["time"] = runtime
            
            y = yv[:, i].flatten()
            y_mask = y != -1
            y = y[y_mask]
            if len(y) < 10:
                continue
            p = pmean[:, i].flatten()[y_mask]
            ub_ = ub[:, i].flatten()[y_mask]
            lb_ = lb[:, i].flatten()[y_mask]
            se = score_e[:, i].flatten()[y_mask]
            fpr, tpr, threshold = roc_curve(y, p, drop_intermediate=False)
            optimal_idx = np.argmax(np.abs(tpr - fpr))  # Maximised youden index 
            optimal_threshold = threshold[optimal_idx]
            sa = (p*y) + ((1-p)*(1-y))
            q25e = np.quantile(se, 0.25)
            q75e = np.quantile(se, 0.75)
            q25a = np.quantile(sa, 0.25)
            q75a = np.quantile(sa, 0.75)
            outside = (ub_ < optimal_threshold) | (lb_ > optimal_threshold)
            store["outside"] = np.sum(outside)
            store["q25e"] = q25e
            store["q75e"] = q75e
            store["q25a"] = q25a
            store["q75a"] = q75a

            correct_ = correct[:, i].flatten().astype(int)[y_mask]

            store["AUC_confidence_a"] = roc_auc_score(y_true=correct_, y_score=sa)
            store["AUC_confidence_e"] = roc_auc_score(y_true=correct_, y_score=se)
            store["NLL"] = np.mean([-(l*np.log(p) + (1-l)*np.log(1-p)) for l,p in zip(y, p)])
            fraction_of_positives, mean_predicted_value = \
                calibration_curve(y, p, n_bins=10)
            store["FRAC_POS"] = fraction_of_positives.tobytes()
            store["MEAN_PRED"] = mean_predicted_value.tobytes()
            store["thresh"] = optimal_threshold

            for metric in metrics:
                store[metric[0] + "_full"] = metric[1](y, p)
                if np.sum(sa < q25a) > 5:
                    p_mu_q25a = p[sa < q25a]
                    labels_q25a = y[sa < q25a]
                    try:
                        store[metric[0] + "_q25a"] = metric[1](labels_q25a, p_mu_q25a)
                    except:
                        store[metric[0] + "_q25a"] = np.nan
                if np.sum(sa > q75a) > 5:
                    p_mu_q75a = p[sa > q75a]
                    labels_q75a = y[sa > q75a]
                    try:
                        store[metric[0] + "_q75a"] = metric[1](labels_q75a, p_mu_q75a)
                    except:
                        store[metric[0] + "_q75a"] = np.nan
                if np.sum(se < q25e) > 5:
                    p_mu_q25e = p[se < q25e]
                    labels_q25e = y[se < q25e]
                    try:
                        store[metric[0] + "_q25e"] = metric[1](labels_q25e, p_mu_q25e)
                    except:
                        store[metric[0] + "_q25e"] = np.nan
                if np.sum(se > q75e) > 5:
                    p_mu_q75e = p[se > q75e]
                    labels_q75e = y[se > q75e]
                    try:
                        store[metric[0] + "_q75e"] = metric[1](labels_q75e, p_mu_q75e)
                    except:
                        store[metric[0] + "_q75e"] = np.nan
                if np.sum(outside) > 5:
                    p_out = p[outside]
                    lab_out = y[outside]
                    try:
                        store[metric[0] + "_conf_out"] = metric[1](lab_out, p_out)
                    except:
                        store[metric[0] + "_conf_out"] = np.nan
    
            outcomes[name] = store

        entries = []
        for name in outcomes:
            o = Outcomes(
                    **outcomes[name]
                )
            entries.append(o)
        session.add_all(entries)
        session.commit()
