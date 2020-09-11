import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn import metrics
from sklearn import preprocessing
import lightgbm as lgb
# Suppress warning
import warnings
warnings.filterwarnings("ignore")
import gc

sub = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')

train=pd.read_csv('../input/ieee-feature-engineering-simple/train_processed_simple.csv')
test=pd.read_csv('../input/ieee-feature-engineering-simple/test_processed_simple.csv')
#all_data=pd.concat([train,test],axis=0)
#del train,test
gc.collect()

train['groups'] = train['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
train['groups'] = (train['groups'].dt.year-2017)*12 + train['groups'].dt.month
#train=train[train['groups']!=12]
#train.reset_index()
target=train['isFraud']
train.drop(['isFraud'],axis=1,inplace=True)
test.drop(['isFraud'],axis=1,inplace=True)

n_fold=6  ########
folds = GroupKFold(n_splits=n_fold)
train['groups'] = train['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
train['groups'] = (train['groups'].dt.year-2017)*12 + train['groups'].dt.month
train=train[train['groups']!=12]

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
pred = np.zeros(test.shape[0])
oof = np.zeros(train.shape[0])
score = []

split_groups = train['groups']
train.drop(['groups','TransactionDT','D15'],axis=1,inplace=True)
test.drop(['TransactionDT','D15'],axis=1,inplace=True)
imp=np.zeros(test.shape[1])
for tra_ind,val_ind in folds.split(train,target,groups=split_groups):
    tra=lgb.Dataset(train.iloc[tra_ind],label=target[tra_ind])
    val=lgb.Dataset(train.iloc[val_ind],label=target[val_ind])
    gc.collect()
    clf=lgb.train(params,tra,num_boost_round =2000,early_stopping_rounds = 100,valid_sets=[tra,val], verbose_eval=100)
    imp+=clf.feature_importance(importance_type='gain')/n_fold
    y_valid=clf.predict(train.iloc[val_ind,:])
    pred+=clf.predict(test)/n_fold
    oof[val_ind]=clf.predict(train.iloc[val_ind,:])
    score.append(metrics.roc_auc_score(target[val_ind],y_valid))
print('CV mean:{0:.4f},CV std:{1:.4f}'.format(np.mean(score),np.std(score)))

plt.figure(figsize=(20, 10))
feature_imp=pd.DataFrame(sorted(zip(imp,train.columns)),columns=['Value','Feature'])
sns.barplot(x='Value',y='Feature',data=feature_imp.sort_values(by="Value", ascending=False)[:50])
features_last_50=feature_imp.sort_values(by="Value", ascending=False)[:-50]
features_last_50.to_csv('last_50.csv',index=False)
plt.tight_layout()
plt.show()
oof_df=pd.DataFrame({'lgb_oof':oof})
oof_df.to_csv('lgb_oof.csv',index=False)
sub['isFraud']=pred
sub.to_csv('submission.csv',index=False)