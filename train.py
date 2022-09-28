import numpy as np
import pandas as pd
import torch as T
from app.APN.training.data_manager.dataloader import CardiacDataset,\
     dataloader
from app.APN.training.data_manager.dataloader_balance import \
    CardiacDatasetBalanced, dataloader_balanced
from app.APN.model.Network import network
from app.APN.model.ConcreteNetwork import network as conetwork
from app.APN.model.PostNetwork import post_network
from app.APN.training.train import train
from app.APN.training.validation_fn_balanced import validation
import argparse

# script for training UAN model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Objective')
    parser.add_argument('model', choices=['GVI', 'PN', 'CONCRETE'],
                        help='one of {GVI, PN}')
    args = parser.parse_args()

    # 14490 goes back to 2019-01-29
    valididx = 14490

    X = pd.read_csv("X.csv", index_col=False)
    Xp = pd.read_csv("Xp.csv", index_col=False)
    y = pd.read_csv("y.csv", index_col=False)
    X_train = X.values[:-valididx, :]
    y_train = y.values[:-valididx, :]
    X_val = X.values[-valididx:, :]
    X_valp = Xp.values[-valididx:, :]
    y_val = y.values[-valididx:, :]

    X_ev = pd.read_csv("X_ev.csv", index_col=False).values
    y_ev = pd.read_csv("y_ev.csv", index_col=False).values

    print(X_train.shape, X_valp.shape, X_ev.shape)

    N = []
    for i in range(y.shape[1]):
        o = []
        o.append(np.nansum(y.values[:, i]))
        o.append(y.shape[0] - np.nansum(y.values[:, i]))
        N.append(o)
    N = np.array(N)
    print(N)

    dataset_train_pre = CardiacDatasetBalanced(
        X_train, y_train, x_all=True, p=0.5
        )
    dl_train_pre = dataloader_balanced(dataset_train_pre, batch_size=64)

    dataset_train = CardiacDatasetBalanced(
        X_train, y_train, x_all=True, p=0.5
        )
    dl_train = dataloader_balanced(dataset_train, batch_size=64)

    dataset_val = CardiacDataset(
        X_val, y_val, x_all=True, p=0.1
        )
    dl_val = dataloader(dataset_val, batch_size=64)

    dataset_val_p = CardiacDataset(
        X_valp, y_val, x_all=True, p=0.3
        )
    dl_val_p = dataloader(dataset_val_p, batch_size=64)

    dataset_ev = CardiacDataset(
        X_ev, y_ev, x_all=True, p=0.1
        )
    dl_ev = dataloader(dataset_ev, batch_size=64)

    # state_dict = T.load(
    #     './app/APN/model/saved/run_bce_d2.pt', map_location="cpu")
    if args.model == "GVI":
        model = network(
                X.shape[1],
                y.shape[1],
                16,
                N,
                d_l=4,
                sample=True,
                lr=1e-3,
                regr=1e-6,
            )
        name_str = "final-GVI"
    elif args.model == "PN":
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
        name_str = "final-PN"
    elif args.model == "CONCRETE":
        model = conetwork(
                X.shape[1],
                y.shape[1],
                16,
                N,
                d_l=4,
                sample=True,
                lr=1e-3,
                regr=1e-7,
            )
        name_str = "final-GVI-CON"

    # model.load_state_dict(state_dict)

    train_routine_pre = train(
        model=model,
        trainloader=dl_train_pre,
        valloaders={
            "validation": dl_val,
            "external_validation": dl_ev,
            "partial_validation": dl_val_p,
        },
        outcome_names=list(y),
        limit_steps=None,
        epochs=50,
        device="cpu",
        metric_fn=validation,
        early_stopping=None,
        custom_fn=None,
        logdir='./app/APN/training/logs/network/',
        save_dir='./app/APN/model/saved/',
        save_name=name_str
    )
    train_routine_full = train(
        model=model,
        trainloader=dl_train,
        valloaders={
            "validation": dl_val,
            "external_validation": dl_ev,
            "partial_validation": dl_val_p,
        },
        outcome_names=list(y),
        limit_steps=None,
        epochs=200,
        device="cpu",
        metric_fn=validation,
        early_stopping=None,
        custom_fn=None,
        logdir='./app/APN/training/logs/network/',
        save_dir='./app/APN/model/saved/',
        save_name=name_str
    )

    T.autograd.set_detect_anomaly(False)
    # train_routine_pre.train(mode="mapping")
    # train_routine_full.train(mode="attention")
    train_routine_full.train(mode="all")
