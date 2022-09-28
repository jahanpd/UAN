import optuna
import argparse
import numpy as np
import pandas as pd
import json
import os

from app.APN.training.make_objective import make_objective
from app.APN.training.data_manager.dataloader import CardiacDataset,\
     dataloader
from app.APN.model.Network import network

"""
SCRIPT FOR HYPERPARAMETER OPTIMIZATION

"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Objective')
    parser.add_argument('name',
                    help='name for database')
    parser.add_argument('device',
                    help='"cuda" for gpu and "cpu" for cpu')
    args = parser.parse_args()
    
    storage = "sqlite:///database/" + args.name + ".db"
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name=args.name, 
        load_if_exists=True
        )

    #### IMPORT FILES FOR MODEL AND DATASET BUILDING 

    validation = 14490

    X = pd.read_csv("X.csv", index_col=False)
    Xp = pd.read_csv("Xp.csv", index_col=False)
    y = pd.read_csv("y.csv", index_col=False)
    X_train = X.values[:-validation, :]
    y_train = y.values[:-validation, :]
    X_val = X.values[-validation:, :]
    X_valp = Xp.values[-validation:, :]
    y_val = y.values[-validation:, :]

    X_ev = pd.read_csv("X_ev.csv", index_col=False).values
    y_ev = pd.read_csv("y_ev.csv", index_col=False).values

    N = []
    for i in range(y.shape[1]):
        o = []
        o.append(np.nansum(y.values[:, i]))
        o.append(y.shape[0] - np.nansum(y.values[:, i]))
        N.append(o)
    N = np.array(N)
    print(N)

    dataset_train = CardiacDataset(
        X_train, y_train, x_all=False, p=0.5
        )
    dl_train = dataloader(dataset_train, batch_size=64)

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

    objective = make_objective(
        features=X.shape[1], 
        outcomes=y.shape[1], 
        N=N, 
        train_dataset=dl_train, 
        ev=dl_ev,
        outcome_names=list(y),
        device=args.device
        )

    study.optimize(objective, n_trials=100)