#!/bin/sh
# rm database/xgboost.db
# cd app/APN/training
# python database_create.py xgboost
# cd ../../..
# python cross_validate.py xgboost xgb
# python cross_validate.py xgboost xgb
# python cross_validate.py xgboost xgb
# python cross_validate.py xgboost xgb

# rm database/lr.db
# cd app/APN/training
# python database_create.py lr
# cd ../../..
# python cross_validate.py lr lr
# python cross_validate.py lr lr
# python cross_validate.py lr lr
# python cross_validate.py lr lr

rm database/attention.db
cd app/APN/training
python database_create.py attention
cd ../../..
python cross_validate.py attention attention
python cross_validate.py attention attention
python cross_validate.py attention attention
python cross_validate.py attention attention

# rm database/attention_mha.db
# cd app/APN/training
# python database_create.py attention_mha
# cd ../../..
# python cross_validate.py attention_mha attention --mha
# python cross_validate.py attention_mha attention --mha
# python cross_validate.py attention_mha attention --mha
# python cross_validate.py attention_mha attention --mha

rm database/apn.db
cd app/APN/training
python database_create.py apn
cd ../../..
python cross_validate.py apn attention --posterior
python cross_validate.py apn attention --posterior
python cross_validate.py apn attention --posterior
python cross_validate.py apn attention --posterior

# rm database/bayes.db
# cd app/APN/training
# python database_create.py bayes
# cd ../../..
# python cross_validate.py bayes attention --bayes_attention
# python cross_validate.py bayes attention --bayes_attention
# python cross_validate.py bayes attention --bayes_attention
# python cross_validate.py bayes attention --bayes_attention

### run with imputation
# rm database/xgboost_mean.db
# cd app/APN/training
# python database_create.py xgboost_mean
# cd ../../..
# python cross_validate.py xgboost_mean xgb --impute mean
# python cross_validate.py xgboost_mean xgb --impute mean
# python cross_validate.py xgboost_mean xgb --impute mean
# python cross_validate.py xgboost_mean xgb --impute mean

# rm database/xgboost_mice.db
# cd app/APN/training
# python database_create.py xgboost_mice
# cd ../../..
# python cross_validate.py xgboost_mice xgb --impute mice
# python cross_validate.py xgboost_mice xgb --impute mice
# python cross_validate.py xgboost_mice xgb --impute mice
# python cross_validate.py xgboost_mice xgb --impute mice

# rm database/xgboost_knn.db
# cd app/APN/training
# python database_create.py xgboost_knn
# cd ../../..
# python cross_validate.py xgboost_knn xgb --impute knn
# python cross_validate.py xgboost_knn xgb --impute knn
# python cross_validate.py xgboost_knn xgb --impute knn
# python cross_validate.py xgboost_knn xgb --impute knn

LR
rm database/lr_mean.db
cd app/APN/training
python database_create.py lr_mean
cd ../../..
python cross_validate.py lr_mean lr --impute mean
python cross_validate.py lr_mean lr --impute mean
python cross_validate.py lr_mean lr --impute mean
python cross_validate.py lr_mean lr --impute mean

rm database/lr_mice.db
cd app/APN/training
python database_create.py lr_mice
cd ../../..
python cross_validate.py lr_mice lr --impute mice
python cross_validate.py lr_mice lr --impute mice
python cross_validate.py lr_mice lr --impute mice
python cross_validate.py lr_mice lr --impute mice

# rm database/lr_knn.db
# cd app/APN/training
# python database_create.py lr_knn
# cd ../../..
# python cross_validate.py lr_knn lr --impute knn
# python cross_validate.py lr_knn lr --impute knn
# python cross_validate.py lr_knn lr --impute knn
# python cross_validate.py lr_knn lr --impute knn