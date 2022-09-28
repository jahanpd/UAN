import numpy as np
from app.APN.model.Network import network
from app.APN.training.train import train
from app.APN.training.validation_fn_opt import validation

def make_objective(
    features, 
    outcomes, 
    train_dataset, 
    N, 
    ev,
    outcome_names,
    device
    ):
    def objective(trial):

        d_m = trial.suggest_int('d_m', 2, 64)
        regr = trial.suggest_uniform('regr', 1e-10, 1e-2)
        regr_a = trial.suggest_uniform('regr_a', 1e-10, 1e-2)
 
        kwargs = {
        'features': features,  # number features
        'outcomes': outcomes,  # number outcomes
        'd_m':d_m,
        'N':N,
        'sample':True,
        'lr':1e-3,
        'scheduler':None,
        'regr':1e-4,
        'regr_a':regr_a
        }

        model = network(**kwargs)
        train_routine = train(
            model=model,
            trainloader=train_dataset,
            valloaders=None,
            outcome_names=outcome_names,
            limit_steps=None,
            epochs=100,
            device=device,
            logdir='./app/APN/training/logs/hyperparameter/',
            save_dir='./app/APN/model/saved/',
            save_name="hyperparameter"
        )
        train_routine.train()
        auc  = validation(
            model=model,
            dl=ev,
            dl_name="external_validation",
            outcome_names=outcome_names,
            epoch=100,
            device=device
        )
        return auc
        
    return objective