# Uncertainty-aware attention network and CardiacML

A novel, interpretable, and uncertainty-aware deep learning model for predicting postoperative outcomes in cardiac surgery.
The model is made available for research purposes at www.cardiac-ml.com.

## Experiments

The model is trained on the [ANZSCTS National Database](https://anzscts.org/database/) with external validation on a cardiac surgery subset of the [MIMIC III database](https://physionet.org/content/mimiciii/1.4/).

The UAN is an uncertainty aware model and so benchmarking is compared against a Bayesian logistic regression model and an ensemble of XGBoost models.

## Application

The CardiacML app is built using a Flask backend and is in in the app directory along with the model.
