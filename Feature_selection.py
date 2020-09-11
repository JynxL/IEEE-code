import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold,KFold
from sklearn import metrics
from sklearn import preprocessing
import lightgbm as lgb
# Suppr warning
import warnings
warnings.filterwarnings("ignore")
import gc
import  datetime
from Untils import *

def permutation_importance(X, y, model):
    perm = {}
    y_true = model.predict(X)
    baseline= roc_auc_score(y, y_true)
    for cols in X.columns:
        value = X[cols].copy()
        X[cols] = np.random.permutation(X[cols])
        y_true = model.predict(X)
        perm[cols] = roc_auc_score(y, y_true) - baseline
        X[cols] = value
    return perm

START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
sub = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
train=pd.read_csv('../input/ieee-feature-engineering/train_processed.csv')
#test=pd.read_csv('../input/ieee-feature-engineering/test_processed.csv')
#all_data=pd.concat([train,test],axis=0)
#del train,test
gc.collect()

train['groups'] = train['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
train['groups'] = (train['groups'].dt.year-2017)*12 + train['groups'].dt.month
#train=train[train['groups']!=12]
#train.reset_index()

#use last month as validation
train.drop(['TransactionDT'],axis=1,inplace=True)
val=train.loc[(train['groups']==17)]
train=train.loc[(train['groups']<=16)]
train_y=train['isFraud']
val_y=val['isFraud']
del train['groups'],val['groups']
del train['isFraud'],val['isFraud']

params = {'num_leaves': 256,  ###256###
          'min_child_samples': 79,
          'objective': 'binary',
          # 'max_depth': 13,#
          'learning_rate': 0.03,
          "boosting_type": "gbdt",
          "subsample_freq": 3,
          # "subsample": 0.78,#
          "bagging_seed": 11,
          'bagging_fraction': 0.5,
          'feature_fraction': 0.38,
          'max_depth': -1,  ####-1####
          'reg_alpha': 1.6,
          'reg_lambda': 1,
          "metric": 'auc',
          "verbosity": 0,
          'reg_alpha': 0.3,
          'reg_lambda': 0.64,
          'min_data_in_leaf': 150,  ####150####
          "verbosity": -1

          # 'categorical_feature': cat_cols
          }

###fit model###
train=lgb.Dataset(train,label=train_y)
clf=lgb.train(params,train,num_boost_round =500, verbose_eval=100)

####do test##
selections=permutation_importance(val, val_y, clf)
s=pd.DataFrame.from_dict(selections,orient='index')
s.index.name='feature'
s.reset_index(inplace=True)
s.rename(columns={1:'value'})
s.to_csv('feature_scores.csv',index=False)
