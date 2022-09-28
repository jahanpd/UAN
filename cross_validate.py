import os
os.chdir('./')
print("Current working directory: {0}".format(os.getcwd()))


import datetime
import numpy as np
import pandas as pd
import torch as T
import argparse
import xgboost as xgb
from app.LogisticRegression.model.LR import BayesianLR
from app.LogisticRegression.training.validation_fn_core import\
    validation_sample
from app.APN.training.train import train
from app.APN.training.validation_fn_crossval import\
    validation_cv, validation_cv_bayes, validation_cv_mha
from app.APN.training.data_manager.dataloader_balance import\
    CardiacDatasetBalanced, dataloader_balanced
from app.APN.training.data_manager.dataloader import CardiacDataset,\
     dataloader
from app.APN.model.Network import network
from app.APN.model.ConcreteNetwork import network as conetwork
from app.APN.model.PostNetwork import post_network
from app.APN.model.BayesNetwork import bayes_network
from app.APN.model.EnsembleNetwork import mha_network
from app.APN.training.database_create import Outcomes
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sklearn.metrics import\
    roc_auc_score, brier_score_loss, recall_score, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from scipy.special import expit


class empty_impute:
    def __init__(self):
        pass

    def transform(self, x):
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Objective')
    parser.add_argument('name',
                        help='name for database')
    parser.add_argument('model', choices=['attention', 'xgb', 'ngb', 'lr'],
                        help='one of {attention, xgb, ngb, lr, apn}')
    parser.add_argument('--posterior', action='store_true',
                        help='posterior network')
    parser.add_argument('--concrete', action='store_true',
                        help='concrete mask')
    parser.add_argument('--mha', action='store_true',
                        help='multihead attention ensemble model')
    parser.add_argument('--impute', choices=['mean', 'mice', 'knn'],
                        help='impute with one of {mean, mice, knn}')
    args = parser.parse_args()

    database = "sqlite:///database/" + args.name + ".db"

    ## CONNECT TO DB
    engine = create_engine(database)
    Session = sessionmaker(bind=engine)
    session = Session()

    # set parameters
    sample_size = 50000

    # prep data
    # 14490 goes back to 2019-01-29
    validation = 14490

    X = pd.read_csv("X.csv", index_col=False)
    Xp = pd.read_csv("Xp.csv", index_col=False)
    y = pd.read_csv("y.csv", index_col=False)

    X = X.replace(-1, np.nan)
    Xp = Xp.replace(-1, np.nan)
    y = y.replace(-1, np.nan)

    X_train = X.values[:-validation, :]
    Xp_train = Xp.values[:-validation, :]

    y_train = y.values[:-validation, :]

    # define imputer
    if args.impute is not None:
        if args.impute == "mean":
            get_imputer = lambda: SimpleImputer(missing_values=np.nan, strategy='mean')
        if args.impute == "mice":    
            get_imputer = lambda: IterativeImputer(max_iter=10, random_state=0, sample_posterior=True)
        if args.impute == "knn":
            get_imputer = lambda: KNNImputer(n_neighbors=2, weights="uniform")
    else:
        get_imputer = lambda: empty_impute()

    X_val = X.values[-validation:, :] # temporal validation
    X_valp = Xp.values[-validation:, :]
    y_val = y.values[-validation:, :]

    X_ev = pd.read_csv("X_ev.csv", index_col=False)
    y_ev = pd.read_csv("y_ev.csv", index_col=False)

    X_ev = X_ev.replace(-1, np.nan)
    X_ev = pd.DataFrame(X_ev, columns=list(X))
    y_ev = y_ev.replace(-1, np.nan)

    outcomes = list(y)

    runtime = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    if args.model != 'attention':
        # subset cols to include for non-attention models
        mask = ~np.all(np.isnan(X_ev.values), axis=0)
        cols = list(np.array(list(X_ev))[mask])

        X_train_ = X_train[:, mask]
        X_val_ = X_val[:, mask]
        X_ev_ = X_ev[cols].values

        for i, outcome in enumerate(outcomes):
            # prep data based on outcomes with full is all features and i is feature subset
            X_train_full = pd.DataFrame(X_train, columns=list(X))
            X_train_full[outcome] = y_train[:, i]
            X_train_full.dropna(subset=[outcome], inplace=True)
            y_train_full = X_train_full.pop(outcome)
            X_train_full = X_train_full.values
            y_train_full = y_train_full.values

            X_val_full = pd.DataFrame(X_val, columns=list(X))
            X_val_full[outcome] = y_val[:, i]
            X_val_full.dropna(subset=[outcome], inplace=True)
            y_val_full = X_val_full.pop(outcome)
            X_val_full = X_val_full.values
            y_val_full = y_val_full.values

            X_train_i = pd.DataFrame(X_train_, columns=cols)
            X_train_i[outcome] = y_train[:, i]
            X_train_drop = X_train_i.dropna()
            X_train_i.dropna(subset=[outcome], inplace=True)
            y_train_drop = X_train_drop.pop(outcome)
            X_train_drop = X_train_drop.values
            y_train_drop = y_train_drop.values
            y_train_i = X_train_i.pop(outcome)
            X_train_i = X_train_i.values
            y_train_i = y_train_i.values

            X_val_i = pd.DataFrame(X_val_, columns=cols)
            X_val_i[outcome] = y_val[:, i]
            X_val_drop = X_val_i.dropna()
            X_val_i.dropna(subset=[outcome], inplace=True)
            y_val_drop = X_val_drop.pop(outcome)
            X_val_drop = X_val_drop.values
            y_val_drop = y_val_drop.values
            y_val_i = X_val_i.pop(outcome)
            X_val_i = X_val_i.values
            y_val_i = y_val_i.values

            if outcome == "MORT30":
                X_ev_i = pd.DataFrame(X_ev_, columns=cols)
                X_ev_i[outcome] = y_ev.values[:, i]
                X_ev_drop = X_ev_i.dropna()
                y_ev_drop = X_ev_drop.pop(outcome)
                X_ev_drop = X_ev_drop.values
                y_ev_drop = y_ev_drop.values
                y_ev_i = X_ev_i.pop(outcome)
                X_ev_i = X_ev_i.values
                y_ev_i = y_ev_i.values

            # prep split train idx into k folds
            rows, _ = X_train_i.shape
            rowsdrop, _ = X_train_drop.shape
            rowsfull, _ = X_train_full.shape
            mask1 = y_train_i == 1
            mask1drop = y_train_drop == 1
            mask1full = y_train_full == 1
            mask0 = y_train_i == 0
            mask0drop = y_train_drop == 0
            mask0full = y_train_full == 0
            idx1 = np.arange(rows)[mask1]
            np.random.shuffle(idx1)
            idx1drop = np.arange(rowsdrop)[mask1drop]
            np.random.shuffle(idx1drop)
            idx1full = np.arange(rowsfull)[mask1full]
            np.random.shuffle(idx1full)
            idx0 = np.arange(rows)[mask0]
            np.random.shuffle(idx0)
            idx0drop = np.arange(rowsdrop)[mask0drop]
            np.random.shuffle(idx0drop)
            idx0full = np.arange(rowsfull)[mask0full]
            np.random.shuffle(idx0full)
            # split
            idx1 = np.array_split(idx1, 5)
            idx1drop = np.array_split(idx1drop, 5)
            idx1full = np.array_split(idx1full, 5)
            idx0 = np.array_split(idx0, 5)
            idx0drop = np.array_split(idx0drop, 5)
            idx0full = np.array_split(idx0full, 5)

            for s in range(5):
                # rebalancing logic
                test_idx = np.concatenate([idx1[s], idx0[s]])
                test_idx_drop = np.concatenate([idx1drop[s], idx0drop[s]])
                test_idx_full = np.concatenate([idx1full[s], idx0full[s]])
                train_idx1 = np.concatenate(idx1[:s] + idx1[s+1:])
                train_idx0 = np.concatenate(idx0[:s] + idx0[s+1:])
                train_idx1drop = np.concatenate(idx1drop[:s] + idx1drop[s+1:])
                train_idx0drop = np.concatenate(idx0drop[:s] + idx0drop[s+1:])
                train_idx1full = np.concatenate(idx1full[:s] + idx1full[s+1:])
                train_idx0full = np.concatenate(idx0full[:s] + idx0full[s+1:])
                # balance training set
                np.random.seed(1+s)
                train_idx1 = np.random.choice(
                    train_idx1, size=sample_size, replace=True
                )
                np.random.seed(1+s)
                train_idx0 = np.random.choice(
                    train_idx0, size=sample_size, replace=True
                )
                np.random.seed(50+s)
                train_idx1drop = np.random.choice(
                    train_idx1drop, size=sample_size, replace=True
                )
                np.random.seed(50+s)
                train_idx0full = np.random.choice(
                    train_idx0full, size=sample_size, replace=True
                )
                np.random.seed(100+s)
                train_idx1full = np.random.choice(
                    train_idx1full, size=sample_size, replace=True
                )
                np.random.seed(100+s)
                train_idx0full = np.random.choice(
                    train_idx0full, size=sample_size, replace=True
                )
                train_idx = np.concatenate([train_idx1, train_idx0])
                train_idx_drop = np.concatenate(
                    [train_idx1drop, train_idx0drop])
                train_idx_full = np.concatenate(
                    [train_idx1full, train_idx0full])

                # train and store metrics for each model
                if args.model == 'xgb':

                    params = {
                        'max_depth': 4,
                        'min_child_weight': 3,
                        'objective': 'binary:logistic',
                        'eval_metric': 'auc'
                    }
                    imputer_i = get_imputer()
                    imputer_i.fit(X_train_i[train_idx, :])
                    dtest = xgb.DMatrix(
                        imputer_i.transform(X_train_i[test_idx, :]), label=y_train_i[test_idx])
                    imputer = get_imputer()
                    imputer.fit(X_train_full[test_idx_full, :])
                    dtest_full = xgb.DMatrix(
                        imputer.transform(X_train_full[test_idx_full, :]),
                        label=y_train_full[test_idx_full])
                    dtestdrop = xgb.DMatrix(
                        X_train_drop[test_idx_drop, :],
                        label=y_train_drop[test_idx_drop])
                    dval = xgb.DMatrix(
                        imputer_i.transform(X_val_i), label=y_val_i)
                    dval_full = xgb.DMatrix(
                        imputer.transform(X_val_full), label=y_val_full)
                    dvaldrop = xgb.DMatrix(
                        X_val_drop, label=y_val_drop)
                    if outcome == "MORT30":
                        dev = xgb.DMatrix(imputer_i.transform(X_ev_i), label=y_ev_i)
                        dev_full = xgb.DMatrix(X_ev.values, label=y_ev_i)

                    n_models = 20
                    models = []
                    models_full = []
                    models_drop = []
                    for _ in range(n_models):
                        train_idx_full_sample = np.random.choice(
                            train_idx_full, size=len(train_idx_full))
                        dtrain_full = xgb.DMatrix(
                            imputer.transform(X_train_full[train_idx_full_sample, :]),
                            label=y_train_full[train_idx_full_sample])

                        train_idx_sample = np.random.choice(train_idx, size=len(train_idx))
                        dtrain = xgb.DMatrix(
                            imputer_i.transform(X_train_i[train_idx_sample, :]),
                            label=y_train_i[train_idx_sample])

                        train_idx_drop_sample = np.random.choice(
                            train_idx_drop, size=len(train_idx_drop))
                        dtraindrop = xgb.DMatrix(
                            X_train_drop[train_idx_drop_sample, :],
                            label=y_train_drop[train_idx_drop_sample])

                        model = xgb.train(
                                    params,
                                    dtrain,
                                    num_boost_round=100,
                                    evals=[(dtrain, "Train"), (dtest, "Test")],
                                    early_stopping_rounds=10
                                )
                        models.append(model)

                        model_full = xgb.train(
                                    params,
                                    dtrain_full,
                                    num_boost_round=100,
                                    evals=[(dtrain_full, "Train"),
                                           (dtest_full, "Test")],
                                    early_stopping_rounds=10
                                )
                        models_full.append(model_full)

                        modeldrop = xgb.train(
                                    params,
                                    dtraindrop,
                                    num_boost_round=100,
                                    evals=[(dtraindrop, "Train"),
                                           (dtestdrop, "Test")],
                                    early_stopping_rounds=10
                                )
                        models_drop.append(modeldrop)

                    if outcome == "MORT30":
                        datasets = [('train', dtrain), ('test', dtest),
                                    ('val', dval), ('ev', dev)]
                        datasets_full = [('train', dtrain_full),
                                         ('test', dtest_full),
                                         ('val', dval_full), ('ev', dev_full)]
                        datasetsdrop = [('train', dtraindrop),
                                        ('test', dtestdrop), ('val', dvaldrop),
                                        ('ev', dev)]
                    else:
                        datasets = [('train', dtrain), ('test', dtest),
                                    ('val', dval)]
                        datasets_full = [('train', dtrain_full),
                                         ('test', dtest_full),
                                         ('val', dval_full)]
                        datasetsdrop = [('train', dtraindrop),
                                        ('test', dtestdrop), ('val', dvaldrop)]
                    
                    entries = []
                    for mname, m, ds in [
                        ('xgb_nans', models, datasets),
                        ('xgb_drop', models_drop, datasetsdrop),
                        ('xgb_full', models_full, datasets_full),
                        ]:
                        for dname, data in ds:
                            # define metric function tuples
                            metrics = [
                                ("AUC", lambda l, p: roc_auc_score(y_true=l, y_score=p)),
                                ("SEN", lambda l, p: recall_score(y_true=l, y_pred=np.around(p), pos_label=1)),
                                ("SPEC", lambda l, p: recall_score(y_true=l, y_pred=np.around(p), pos_label=0)),
                                ("BRIER", lambda l, p: brier_score_loss(y_true=l, y_prob=p)),
                                ("ACC", lambda l, p: np.sum(np.around(p) == l) / p.size)
                                ]

                            preds = np.vstack([m_.predict(data) for m_ in m])  # (n_models, preds)
                            p_mu = preds.mean(0)
                            p_std = preds.std(0)
                            labels = data.get_label()


                            fpr, tpr, threshold = roc_curve(labels, p_mu, drop_intermediate=False)
                            optimal_idx = np.argmax(np.abs(tpr - fpr)) # Maximised youden index 
                            optimal_threshold = threshold[optimal_idx]

                            preds_bin = np.array([1.0 if x > optimal_threshold else 0.0 for x in p_mu])
                            correct = (preds_bin == labels.flatten()).astype(int)
                            s_a = (p_mu*labels.flatten()) + ((1-p_mu)*(1-labels.flatten()))
                            s_e = 1/p_std
                            auc_a = roc_auc_score(y_true=correct, y_score=s_a)
                            auc_e = roc_auc_score(y_true=correct, y_score=(s_e))
                            NLL = np.mean([-(l*np.log(p) + (1-l)*np.log(1-p)) for l,p in zip(labels.flatten(), p_mu)])
                            fraction_of_positives, mean_predicted_value = \
                                calibration_curve(labels.flatten(), p_mu, n_bins=10)

                            # confidence stratification
                            p_ub = p_mu + 1.96*p_std
                            p_lb = p_mu - 1.96*p_std
                            q25e = np.quantile(s_e, 0.25)
                            q75e = np.quantile(s_e, 0.75)
                            q25a = np.quantile(s_a, 0.25)
                            q75a = np.quantile(s_a, 0.75)
                            outside = (p_ub < optimal_threshold) | (p_lb > optimal_threshold)
                            store = {
                                "outside":np.sum(outside),
                                "q25e":q25e,
                                "q75e":q75e,
                                "q25a":q25a,
                                "q75a":q75a,
                                "thresh":optimal_threshold
                                }
                            for metric in metrics:
                                store[metric[0] + "_full"] = metric[1](labels, p_mu)
                                if np.sum(s_a < q25a) > 5:
                                    p_mu_q25a = p_mu[s_a < q25a]
                                    labels_q25a = labels[s_a < q25a]
                                    try:
                                        store[metric[0] + "_q25a"] = metric[1](labels_q25a, p_mu_q25a)
                                    except:
                                        store[metric[0] + "_q25a"] = np.nan
                                if np.sum(s_a > q75a) > 5:
                                    p_mu_q75a = p_mu[s_a > q75a]
                                    labels_q75a = labels[s_a > q75a]
                                    try:
                                        store[metric[0] + "_q75a"] = metric[1](labels_q75a, p_mu_q75a)
                                    except:
                                        store[metric[0] + "_q75a"] = np.nan
                                if np.sum(s_e < q25e) > 5:
                                    p_mu_q25e = p_mu[s_e < q25e]
                                    labels_q25e = labels[s_e < q25e]
                                    try:
                                        store[metric[0] + "_q25e"] = metric[1](labels_q25e, p_mu_q25e)
                                    except:
                                        store[metric[0] + "_q25e"] = np.nan
                                if np.sum(s_e > q75e) > 5:
                                    p_mu_q75e = p_mu[s_e > q75e]
                                    labels_q75e = labels[s_e > q75e]
                                    try:
                                        store[metric[0] + "_q75e"] = metric[1](labels_q75e, p_mu_q75e)
                                    except:
                                        store[metric[0] + "_q75e"] = np.nan
                                if np.sum(outside) > 5:
                                    p_out = p_mu[outside]
                                    lab_out = labels[outside]
                                    try:
                                        store[metric[0] + "_conf_out"] = metric[1](lab_out, p_out)
                                    except:
                                        store[metric[0] + "_conf_out"] = np.nan

                            o = Outcomes(
                                    outcome = outcome,
                                    data = dname,
                                    model = mname,
                                    fold = s,
                                    time = runtime,
                                    AUC_confidence_a = auc_a,
                                    AUC_confidence_e = auc_e,
                                    FRAC_POS = fraction_of_positives.tobytes(),
                                    MEAN_PRED = mean_predicted_value.tobytes(),
                                    NLL = NLL,
                                    **store
                                )
                            entries.append(o)
                    session.add_all(entries)
                    session.commit()

                if args.model == 'lr':
                    if args.impute is None:
                        scaler = StandardScaler()
                        scaler.fit(X_train_drop)
                        dataset_train = CardiacDatasetBalanced(
                            scaler.transform(X_train_drop[train_idx_drop, :]), 
                            y_train_drop[train_idx_drop].reshape((-1,1)), x_all=True
                            )
                        dl_train = dataloader_balanced(dataset_train, batch_size=64)
                        dataset_train_ = CardiacDatasetBalanced(
                            scaler.transform(X_train_drop[train_idx_drop, :]), 
                            y_train_drop[train_idx_drop].reshape((-1,1)), x_all=True
                            )
                        dl_train_ = dataloader_balanced(dataset_train_, batch_size=8)

                        dataset_test = CardiacDataset(
                            scaler.transform(X_train_drop[test_idx_drop, :]), 
                            y_train_drop[test_idx_drop].reshape((-1,1)), x_all=True
                            )
                        dl_test = dataloader(dataset_test, batch_size=8)
                        
                        dataset_val = CardiacDataset(
                            scaler.transform(X_val_drop), 
                            y_val_drop.reshape((-1,1)), x_all=True
                            )
                        dl_val = dataloader(dataset_val, batch_size=8)
                        if outcome == "MORT30":
                            dataset_ev = CardiacDataset(
                            scaler.transform(X_ev_drop),
                            y_ev_drop.reshape((-1,1)), x_all=True
                            )
                            dl_ev = dataloader(dataset_ev, batch_size=8)
                    else:
                        imputer = get_imputer()
                        imputer.fit(X_train_i[train_idx, :])

                        scaler = StandardScaler()
                        scaler.fit(X_train_i[train_idx, :])
                        dataset_train = CardiacDatasetBalanced(
                            scaler.transform(imputer.transform(X_train_i[train_idx, :])), 
                            y_train_i[train_idx].reshape((-1,1)), x_all=True
                            )
                        dl_train = dataloader_balanced(dataset_train, batch_size=64)

                        dataset_train_ = CardiacDatasetBalanced(
                            scaler.transform(imputer.transform(X_train_i[train_idx, :])), 
                            y_train_i[train_idx].reshape((-1,1)), x_all=True
                            )
                        dl_train_ = dataloader_balanced(dataset_train_, batch_size=8)

                        dataset_test = CardiacDataset(
                            scaler.transform(imputer.transform(X_train_i[test_idx, :])), 
                            y_train_i[test_idx].reshape((-1,1)), x_all=True
                            )
                        dl_test = dataloader(dataset_test, batch_size=8)
                        
                        dataset_val = CardiacDataset(
                            scaler.transform(imputer.transform(X_val_i[test_idx, :])), 
                            y_val_drop.reshape((-1,1)), x_all=True
                            )
                        dl_val = dataloader(dataset_val, batch_size=8)
                        if outcome == "MORT30":
                            dataset_ev = CardiacDataset(
                            scaler.transform(imputer.transform(X_ev)),
                            y_ev_drop.reshape((-1,1)), x_all=True
                            )
                            dl_ev = dataloader(dataset_ev, batch_size=8)

                    model = BayesianLR(X_train_drop.shape[1])
                    train_routine = train(
                        model=model,
                        trainloader=dl_train,
                        outcome_names=list(y),
                        epochs=50,
                        device="cpu",
                        logdir='./app/LogisticRegression/training/logs/',
                        save_dir='./app/LogisticRegression/model/saved/',
                        save_name="lr" + outcome
                    )
                    T.autograd.set_detect_anomaly(False)
                    train_routine.train()

                    if outcome == "MORT30":
                        datasets = [('train', dl_train_), ('test', dl_test), ('val', dl_val), ('ev', dl_ev)]
                    else:
                        datasets = [('train', dl_train_), ('test', dl_test), ('val', dl_val)]

                    entries = []
                    for dname, ds in datasets:
                        logits_mx, labels = validation_sample(
                            model=model,
                            dl=ds,
                            device="cpu"
                        )
                        
                        # define metric function tuples
                        metrics = [
                            ("AUC", lambda l, p: roc_auc_score(y_true=l, y_score=p)),
                            ("SEN", lambda l, p: recall_score(y_true=l, y_pred=np.around(p), pos_label=1)),
                            ("SPEC", lambda l, p: recall_score(y_true=l, y_pred=np.around(p), pos_label=0)),
                            ("BRIER", lambda l, p: brier_score_loss(y_true=l, y_prob=p)),
                            ("ACC", lambda l, p: np.sum(np.around(p) == l) / p.size)
                            ]

                        logits_mu = logits_mx.mean(0).flatten()
                        logits_std = logits_mx.std(0).flatten()
                        p_mu = expit(logits_mu)

                        fpr, tpr, threshold = roc_curve(labels, p_mu, drop_intermediate=False)
                        optimal_idx = np.argmax(np.abs(tpr - fpr)) # Maximised youden index 
                        optimal_threshold = threshold[optimal_idx]
                        
                        preds_bin = np.array([1.0 if x > optimal_threshold else 0.0 for x in p_mu])
                        correct = (preds_bin == labels.flatten()).astype(int)
                        s_a = (p_mu*labels.flatten()) + ((1-p_mu)*(1-labels.flatten()))
                        s_e = 1/logits_std
                        auc_a = roc_auc_score(y_true=correct, y_score=s_a)
                        auc_e = roc_auc_score(y_true=correct, y_score=(1/logits_std))
                        NLL = np.mean([-(l*np.log(p) + (1-l)*np.log(1-p)) for l,p in zip(labels.flatten(), p_mu)])
                        fraction_of_positives, mean_predicted_value = \
                            calibration_curve(labels.flatten(), p_mu, n_bins=10)
                        
                        # confidence stratification
                        logits_ub = logits_mu + 1.96*logits_std
                        logits_lb = logits_mu - 1.96*logits_std
                        p_ub = expit(logits_ub)
                        p_lb = expit(logits_lb)
                        q25e = np.quantile(s_e, 0.25)
                        q75e = np.quantile(s_e, 0.75)
                        q25a = np.quantile(s_a, 0.25)
                        q75a = np.quantile(s_a, 0.75)
                        outside = (p_ub < optimal_threshold) | (p_lb > optimal_threshold)
                        store = {
                            "outside":np.sum(outside),
                            "q25e":q25e,
                            "q75e":q75e,
                            "q25a":q25a,
                            "q75a":q75a,
                            "thresh":optimal_threshold
                            }
                        for metric in metrics:
                            store[metric[0] + "_full"] = metric[1](labels.flatten(), p_mu)
                            if np.sum(s_a < q25a) > 5:
                                p_mu_q25a = p_mu[s_a < q25a]
                                labels_q25a = labels.flatten()[s_a < q25a]
                                store[metric[0] + "_q25a"] = metric[1](labels_q25a, p_mu_q25a)
                            if np.sum(s_a > q75a) > 5:
                                p_mu_q75a = p_mu[s_a > q75a]
                                labels_q75a = labels.flatten()[s_a > q75a]
                                store[metric[0] + "_q75a"] = metric[1](labels_q75a, p_mu_q75a)
                            if np.sum(s_e < q25e) > 5:
                                p_mu_q25e = p_mu[s_e < q25e]
                                labels_q25e = labels.flatten()[s_e < q25e]
                                store[metric[0] + "_q25e"] = metric[1](labels_q25e, p_mu_q25e)
                            if np.sum(s_e > q75e) > 5:
                                p_mu_q75e = p_mu[s_e > q75e]
                                labels_q75e = labels.flatten()[s_e > q75e]
                                store[metric[0] + "_q75e"] = metric[1](labels_q75e, p_mu_q75e)
                            if np.sum(outside) > 5:
                                p_out = p_mu[outside]
                                lab_out = labels.flatten()[outside]
                                store[metric[0] + "_conf_out"] = metric[1](lab_out, p_out)

                        o = Outcomes(
                                outcome = outcome,
                                data = dname,
                                model = 'LR',
                                fold = s,
                                time = runtime,
                                AUC_confidence_a = auc_a,
                                AUC_confidence_e = auc_e,
                                FRAC_POS = fraction_of_positives.tobytes(),
                                MEAN_PRED = mean_predicted_value.tobytes(),
                                NLL = NLL,
                                **store
                            )
                        entries.append(o)
                    session.add_all(entries)
                    session.commit()
    
    if args.model == "attention":
        N = []
        for i in range(y.shape[1]):
            o = []
            o.append(np.nansum(y.values[:, i]))
            o.append(y.shape[0] - np.nansum(y.values[:, i]))
            N.append(o)
        N = np.array(N)
        print(N)
        # prep split train idx into k folds
        rows, _ = X_train.shape
        idx = np.arange(rows)
        np.random.shuffle(idx)
        # split
        idx = np.array_split(idx, 5)
        for s in range(5):
            if args.posterior:
                model = post_network(
                    X.shape[1],
                    y.shape[1],
                    16,
                    N,
                    d_l=4,
                    sample=True,
                    lr=1e-3,
                    regr=1e-6,
                    )
                device = "cpu"
                all_=False
                valid_fn = validation_cv
            else:
                model = network(
                    X.shape[1],
                    y.shape[1],
                    16,
                    N,
                    d_l=6,
                    sample=True,
                    lr=1e-3,
                    regr=1e-6
                )
                device="cuda"
                all_=False
                valid_fn = validation_cv

            test_idx = idx[s]
            train_idx = np.concatenate(idx[:s] + idx[s+1:])

            imputer = get_imputer()
            imputer.fit(X_train[train_idx, :])
            dataset_train_pre = CardiacDatasetBalanced(
                imputer.transfrom(X_train[train_idx, :]), y_train[train_idx, :], x_all=True, p=0.5
                )
            dl_train_pre = dataloader_balanced(dataset_train_pre, batch_size=64)

            dataset_train = CardiacDatasetBalanced(
                imputer.transform(X_train[train_idx, :]), y_train[train_idx, :], x_all=all_, p=0.5
                )
            dl_train = dataloader_balanced(dataset_train, batch_size=64)

            dataset_test = CardiacDataset(
                imputer.transform(X_train[test_idx, :]), y_train[test_idx, :], x_all=True, p=0.1
                )
            dl_test = dataloader(dataset_test, batch_size=64)

            dataset_val = CardiacDataset(
                imputer.transform(X_val), y_val, x_all=True, p=0.1
                )
            dl_val = dataloader(dataset_val, batch_size=64)

            imputer2 = get_imputer()
            imputer2.fit(Xp_train[train_idx, :])
            dataset_val_p = CardiacDataset(
                imputer2.transform(X_valp), y_val, x_all=True, p=0.3
                )
            dl_val_p = dataloader(dataset_val_p, batch_size=64)

            imputer3 = get_imputer()
            imputer3.fit(X_train_i[train_idx, :])
            dataset_ev = CardiacDataset(
                imputer2.transform(X_ev.values), y_ev.values, x_all=True, p=0.1
                )
            dl_ev = dataloader(dataset_ev, batch_size=64)

            train_routine_pre = train(
                model=model,
                trainloader=dl_train_pre,
                epochs=50,
                outcome_names=list(y),
                device=device,
                logdir='./app/APN/training/logs/crossval/',
                save_dir='./app/APN/model/saved/',
                save_name="crossval_pre"
            )
            train_routine_full = train(
                model=model,
                trainloader=dl_train,
                epochs=200,
                outcome_names=list(y),
                device=device,
                logdir='./app/APN/training/logs/crossval/',
                save_dir='./app/APN/model/saved/',
                save_name="crossval_full"
            )
            T.autograd.set_detect_anomaly(False)
            # train_routine_pre.train(mode="mapping")
            train_routine_full.train(mode="all")

            # check training set metrics
            valid_fn(
                model=model,
                dl=dl_train,
                dl_name='train',
                outcome_names=list(y),
                fold=s,
                runtime=runtime,
                session=session,
                device="cpu"
            )
            valid_fn(
                model=model,
                dl=dl_test,
                dl_name='test',
                outcome_names=list(y),
                fold=s,
                runtime=runtime,
                session=session,
                device="cpu"
            )
            valid_fn(
                model=model,
                dl=dl_val,
                dl_name='val',
                outcome_names=list(y),
                fold=s,
                runtime=runtime,
                session=session,
                device="cpu"
            )
            valid_fn(
                model=model,
                dl=dl_val_p,
                dl_name='val_p',
                outcome_names=list(y),
                fold=s,
                runtime=runtime,
                session=session,
                device="cpu"
            )
            valid_fn(
                model=model,
                dl=dl_ev,
                dl_name='ev',
                outcome_names=list(y),
                fold=s,
                runtime=runtime,
                session=session,
                device="cpu"
            )
