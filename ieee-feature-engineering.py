


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
import gc
import datetime
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from Untils import *

dates_range = pd.date_range(start='2017-10-01', end='2019-01-01')
us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())


# read files
folder_path = '../input/'
print('Loading data...')

train_identity = pd.read_csv(f'{folder_path}train_identity.csv', index_col='TransactionID')
print('\tSuccessfully loaded train_identity!')

train_transaction = pd.read_csv(f'{folder_path}train_transaction.csv', index_col='TransactionID')
print('\tSuccessfully loaded train_transaction!')

test_identity = pd.read_csv(f'{folder_path}test_identity.csv', index_col='TransactionID')
print('\tSuccessfully loaded test_identity!')

test_transaction = pd.read_csv(f'{folder_path}test_transaction.csv', index_col='TransactionID')
print('\tSuccessfully loaded test_transaction!')

sub = pd.read_csv(f'{folder_path}sample_submission.csv')
print('\tSuccessfully loaded sample_submission!')

print('Data was successfully loaded!\n')




def id_split(dataframe):
    #split id columns, mapping device to brands
    dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0]
    dataframe['device_version'] = dataframe['DeviceInfo'].str.split('/', expand=True)[1]

    dataframe['OS_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[0]
    dataframe['version_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[1]

    dataframe['browser_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[0]
    dataframe['version_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[1]

    dataframe['screen_width'] = dataframe['id_33'].str.split('x', expand=True)[0]
    dataframe['screen_height'] = dataframe['id_33'].str.split('x', expand=True)[1]

    dataframe['id_34'] = dataframe['id_34'].str.split(':', expand=True)[1]
    dataframe['id_23'] = dataframe['id_23'].str.split(':', expand=True)[1]

    dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
    dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
    dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
    dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
    dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
    dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
    dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
    dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

    dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"
    dataframe['had_id'] = 1
    gc.collect()
    
    return dataframe


train_identity = id_split(train_identity)
test_identity = id_split(test_identity)

#merge data
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
del train_identity, train_transaction, test_identity, test_transaction
gc.collect()

# drop some columns with too many nulls
one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]
one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]
many_null_cols = [col for col in train.columns if train[col].isnull().sum() / train.shape[0] > 0.9]
many_null_cols_test = [col for col in test.columns if test[col].isnull().sum() / test.shape[0] > 0.9]
cols_to_drop=list(set(one_value_cols+one_value_cols_test))
#cols_to_drop.remove('id_02')
print('{} features are going to be dropped for being useless'.format(len(cols_to_drop)))
print(cols_to_drop)
#train = train.drop(cols_to_drop, axis=1) drop in the end####
#test = test.drop(cols_to_drop, axis=1)

# reduce memory usage
reduce_cols=list(test.columns)
reduce_cols.remove('TransactionAmt')
reduce_cols.remove('D1')
train=reduce_mem_usage(train)
test=reduce_mem_usage(test)



train['Transaction_ID']=train.index
test['Transaction_ID']=test.index
###fill cards na
i_cols = ['card2','card3','card4','card5']
full_df = pd.concat([train[i_cols], test[i_cols]])

## We will find best match for nan values and fill with it
for col in i_cols:
    temp_df = full_df.groupby(['card1',col])[col].agg(['count']).reset_index()
    temp_df = temp_df.sort_values(by=['card1','count'], ascending=False).reset_index(drop=True)
    del temp_df['count']
    temp_df = temp_df.drop_duplicates(keep='first').reset_index(drop=True)
    temp_df.index = temp_df['card1'].values
    temp_df = temp_df[col].to_dict()
    full_df[col] = np.where(full_df[col].isna(), full_df['card1'].map(temp_df), full_df[col])
    
    
i_cols = ['card2','card3','card4','card5']
for col in i_cols:
    train[col] = full_df[full_df['Transaction_ID'].isin(train['Transaction_ID'])][col].values
    test[col] = full_df[full_df['Transaction_ID'].isin(test['Transaction_ID'])][col].values
del full_df
del train['Transaction_ID'],test['Transaction_ID']


# clean card1#####
#i_cols = ['card1']

#for col in i_cols: 
    #valid_card = pd.concat([train[[col]], test[[col]]])
    #valid_card = valid_card[col].value_counts()
    #valid_card = valid_card[valid_card>2]
    #valid_card = list(valid_card.index)

    #train[col] = np.where(train[col].isin(test[col]), train[col], np.nan)
    #test[col]  = np.where(test[col].isin(train[col]), test[col], np.nan)

    #train[col] = np.where(train[col].isin(valid_card), train[col], np.nan)
    #test[col]  = np.where(test[col].isin(valid_card), test[col], np.nan)#




# clean emails
emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}
us_emails = ['gmail', 'net', 'edu']

for c in ['P_emaildomain', 'R_emaildomain']:
    train[c + '_bin'] = train[c].map(emails)
    test[c + '_bin'] = test[c].map(emails)
    
    train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])
    test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])
    
    train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
    test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')





####combine train test###
test['isFraud']=-1
all_data=pd.concat([train,test],axis=0)
del train,test
gc.collect()



#calculate some stats

all_data['TransactionAmt_to_mean_card1'] = all_data['TransactionAmt'] / all_data.groupby(['card1'])['TransactionAmt'].transform('mean')
all_data['TransactionAmt_to_mean_card4'] = all_data['TransactionAmt'] / all_data.groupby(['card4'])['TransactionAmt'].transform('mean')
all_data['TransactionAmt_to_std_card1'] = all_data['TransactionAmt'] / all_data.groupby(['card1'])['TransactionAmt'].transform('std')
all_data['TransactionAmt_to_std_card4'] = all_data['TransactionAmt'] / all_data.groupby(['card4'])['TransactionAmt'].transform('std')

all_data['id_02_to_mean_card1'] = all_data['id_02'] / all_data.groupby(['card1'])['id_02'].transform('mean')
all_data['id_02_to_mean_card4'] = all_data['id_02'] / all_data.groupby(['card4'])['id_02'].transform('mean')
all_data['id_02_to_std_card1'] = all_data['id_02'] / all_data.groupby(['card1'])['id_02'].transform('std')
all_data['id_02_to_std_card4'] = all_data['id_02'] / all_data.groupby(['card4'])['id_02'].transform('std')


# identify user !

####card identifier#####
all_data['uid'] = all_data['card1'].astype(str)+'_'+all_data['card2'].astype(str)+'_'
+all_data['card3'].astype(str)+'_'+all_data['card4'].astype(str)

all_data['uid2'] = all_data['uid'].astype(str)+'_'+all_data['addr1'].astype(str)+'_'+all_data['addr2'].astype(str)

all_data['uid3'] = all_data['uid'].astype(str)+'_'+all_data['ProductCD'].astype(str)###new####


####eng uid4#####
all_data['DaysFromStart'] = np.round(all_data['TransactionDT']/(60*60*24),0)
all_data['diff_date']=all_data['D1']-all_data['DaysFromStart']
all_data['uid4']=all_data['uid'].astype(str)+'_'+all_data['card5'].astype(str)+'_'+all_data['card6'].astype(str)+'_'+all_data['diff_date'].astype(str)

del all_data['DaysFromStart']
del all_data['diff_date']

def fast_mode(df, key_cols, value_col): # copied from stack overflow
    """ 
    Calculate a column mode, by group, ignoring null values. 

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame over which to calcualate the mode. 
    key_cols : list of str
        Columns to groupby for calculation of mode.
    value_col : str
        Column for which to calculate the mode. 

    Return
    ------ 
    pandas.DataFrame
        One row for the mode of value_col per key_cols group. If ties, 
        returns the one which is sorted first. 
    """
    return (df.groupby(key_cols + [value_col]).size() 
              .to_frame('counts').reset_index() 
              .sort_values('counts', ascending=False) 
              .drop_duplicates(subset=key_cols)).drop(columns='counts')

#fill addr1 na values with mode
all_data.loc[all_data.addr1.isnull(), 'addr1'] = all_data.uid4.map(fast_mode(all_data, ['uid4'], 'addr1').set_index('uid4').addr1)


all_data['uid4']=all_data['uid4'].astype(str)+'_'+all_data['addr1'].astype(str)

# agg by id cols
i_cols = ['card2','card3','card5','uid','uid2','uid3','uid4']
for col in i_cols:
    for agg_type in ['mean', 'std']:
        new_col_name = col+'_TransactionAmt_'+agg_type
        all_data[new_col_name] = all_data.groupby([col])['TransactionAmt'].transform(agg_type)
#all_data.drop(['uid','uid2'],axis=1,inplace=True)

#agg distance
i_cols = ['addr1','addr2','card3']

for col in i_cols:
    for agg_type in ['mean', 'std']:
        new_col_name = col+'_dist1_'+agg_type
        all_data[new_col_name] = all_data.groupby([col])['dist1'].transform(agg_type)
for col in i_cols:
    for agg_type in ['mean', 'std']:
        new_col_name = col+'_dist2_'+agg_type
        all_data[new_col_name] = all_data.groupby([col])['dist2'].transform(agg_type)
#agg d3
i_cols = ['uid4']

for col in i_cols:
    for agg_type in ['mean', 'std']:
        new_col_name = col+'_D3_'+agg_type
        all_data[new_col_name] = all_data.groupby([col])['D3'].transform(agg_type)
for col in i_cols:
    for agg_type in ['mean', 'std']:
        new_col_name = col+'_C1_'+agg_type
        all_data[new_col_name] = all_data.groupby([col])['C1'].transform(agg_type)

####v258 agg by uid4##
i_cols = ['uid4']
for col in i_cols:
    for agg_type in ['mean', 'std']:
        new_col_name = col+'_V258_'+agg_type
        all_data[new_col_name] = all_data.groupby([col])['V258'].transform(agg_type)
        
###dist by uid4###NEW
for col in i_cols:
    for agg_type in ['mean', 'std']:
        new_col_name = col+'_dist1_'+agg_type
        all_data[new_col_name] = all_data.groupby([col])['dist1'].transform(agg_type)
for col in i_cols:
    for agg_type in ['mean', 'std']:
        new_col_name = col+'_dist2_'+agg_type
        all_data[new_col_name] = all_data.groupby([col])['dist2'].transform(agg_type)
#####d2 d15 by uid4 ####NEW
for col in i_cols:
    for agg_type in ['mean', 'std']:
        new_col_name = col+'_D2_'+agg_type
        all_data[new_col_name] = all_data.groupby([col])['D2'].transform(agg_type)
for col in i_cols:
    for agg_type in ['mean', 'std']:
        new_col_name = col+'_D15_'+agg_type
        all_data[new_col_name] = all_data.groupby([col])['D15'].transform(agg_type)
###D4 by uid4 NEW
for col in i_cols:
    for agg_type in ['mean', 'std']:
        new_col_name = col+'_D4_'+agg_type
        all_data[new_col_name] = all_data.groupby([col])['D4'].transform(agg_type) 
###d10 d11 NEW##

for col in i_cols:
    for agg_type in ['mean', 'std']:
        new_col_name = col+'_D10_'+agg_type
        all_data[new_col_name] = all_data.groupby([col])['D10'].transform(agg_type)
for col in i_cols:
    for agg_type in ['mean', 'std']:
        new_col_name = col+'_D11_'+agg_type
        all_data[new_col_name] = all_data.groupby([col])['D11'].transform(agg_type)
        

###d3 max min uid4###NEW
i_cols = ['uid4']

for col in i_cols:
    for agg_type in ['max','min','median']:
        new_col_name = col+'_D3_'+agg_type
        all_data[new_col_name] = all_data.groupby([col])['D3'].transform(agg_type)
        
####c13##New
for col in i_cols:
    for agg_type in ['mean', 'std']:
        new_col_name = col+'_C13_'+agg_type
        all_data[new_col_name] = all_data.groupby([col])['C13'].transform(agg_type)


####amt to uid4 mean ###NEW
#all_data['amt_to_uid4_mean']=all_data['TransactionAmt']/all_data['uid4_TransactionAmt_mean']
        
###expanding mean amt uid4####
#all_data['amt_uid4_exp_mean']=all_data.groupby(['uid4'])['TransactionAmt'].transform(lambda x:x.expanding().mean())
        
######cumsum cumsount amt by uid4#####
#all_data['amt_cumsum']=all_data.groupby(['uid4'])['TransactionAmt'].cumsum()
#all_data['amt_cumcount']=all_data.groupby(['uid4'])['TransactionAmt'].cumcount()

        
####diff shift oct)change by uid4###NEW
#all_data['TransactionAmt_shift'] = all_data.groupby(['uid4'])['TransactionAmt'].transform(lambda x:x.shift())
#all_data['TransactionAmt_diff'] = all_data.groupby(['uid4'])['TransactionAmt'].transform(lambda x:x.diff())        
#all_data['C1_pct_change'] = all_data.groupby(['uid4'])['C1'].transform(lambda x:x.pct_change())
#all_data['V258_pct_change'] = all_data.groupby(['uid4'])['V258'].transform(lambda x:x.pct_change())
####diff shift pct forward###
#all_data['TransactionAmt_shift'] = all_data.groupby(['uid4'])['TransactionAmt'].transform(lambda x:x.shift(-1))
#all_data['TransactionAmt_diff'] = all_data.groupby(['uid4'])['TransactionAmt'].transform(lambda x:x.diff(-1))        
#all_data['TransactionAmt_pct_change'] = all_data.groupby(['uid4'])['TransactionAmt'].transform(lambda x:x.pct_change(-1))
#all_data.drop(['uid','uid2'],axis=1,inplace=True)

#####uid emails#####
#all_data['uid_email_number']=all_data.groupby(['uid'])['P_emaildomain'].transform('nunique')
#all_data['uid2_email_number']=all_data.groupby(['uid2'])['P_emaildomain'].transform('nunique')
#all_data['uid_Remail_number']=all_data.groupby(['uid'])['R_emaildomain'].transform('nunique')
#all_data['uid2_Remail_number']=all_data.groupby(['uid2'])['R_emaildomain'].transform('nunique')
#all_data['email_agreement']=(all_data['P_emaildomain']==all_data['R_emaildomain'])



all_data['TransactionAmt_decimal'] = ((all_data['TransactionAmt'] - all_data['TransactionAmt'].astype(int)) * 1000).astype(int)

all_data['Transaction_day_of_week'] = np.floor((all_data['TransactionDT'] / (3600 * 24) - 1) % 7)

# more time features
START_DATE = '2017-11-30'
startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
all_data['DT'] = all_data['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))

all_data['DT_M'] = ((all_data['DT'].dt.year-2017)*12 + all_data['DT'].dt.month).astype(np.int8)
all_data['DT_W'] = ((all_data['DT'].dt.year-2017)*52 + all_data['DT'].dt.weekofyear).astype(np.int8)
all_data['DT_D'] = ((all_data['DT'].dt.year-2017)*365 + all_data['DT'].dt.dayofyear).astype(np.int16)
    
#all_data['DT_hour'] = (all_data['DT'].dt.hour).astype(np.int8)
all_data['DT_hour'] =np.floor(all_data['TransactionDT']/3600)  ####new###
all_data['DT_day_week'] = (all_data['DT'].dt.dayofweek).astype(np.int8)
all_data['DT_day_month'] = (all_data['DT'].dt.day).astype(np.int8)

all_data['is_december'] = all_data['DT'].dt.month
all_data['is_december'] = (all_data['is_december']%12==0).astype(np.int8) ###correction NEW###

# Holidays
all_data['is_holiday'] = (all_data['DT'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)


#feature intercations

for feature in [ 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain', 'P_emaildomain__C2', 
                'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 'addr1__card1']:
    f1, f2 = feature.split('__')
    all_data[feature] = all_data[f1].astype(str) + '_' + all_data[f2].astype(str)



    
# count encoding
for feature in ['card1','card2','card3','card5',
          'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
          'D1','D2','D9',
          'addr1','addr2',
          'dist1','dist2',
          'P_emaildomain', 'R_emaildomain','uid4']:
    all_data[feature + '_count_full'] = all_data[feature].map(all_data[feature].value_counts(dropna=False))###uid3 count new###




# hour of day
def make_hour_feature(df, tname='TransactionDT'):
    hours = df[tname] / (3600)        
    encoded_hours = np.floor(hours) % 24
    return encoded_hours
all_data['hour']=make_hour_feature(all_data)



# process d  feature###
def values_normalization(dt_df, periods, columns):
    for period in periods:
        for col in columns:
            new_col = col +'_'+ period
            dt_df[col] = dt_df[col].astype(float)  

            temp_min = dt_df.groupby([period])[col].agg(['min']).reset_index()
            temp_min.index = temp_min[period].values
            temp_min = temp_min['min'].to_dict()

            temp_max = dt_df.groupby([period])[col].agg(['max']).reset_index()
            temp_max.index = temp_max[period].values
            temp_max = temp_max['max'].to_dict()

            dt_df['temp_min'] = dt_df[period].map(temp_min)
            dt_df['temp_max'] = dt_df[period].map(temp_max)

            dt_df[new_col+'_min_max'] = (dt_df[col]-dt_df['temp_min'])/(dt_df['temp_max']-dt_df['temp_min'])

            del dt_df['temp_min'],dt_df['temp_max']
    return dt_df
###d15 by day###
periods = ['DT_D']
all_data = values_normalization(all_data, periods, ['D15'])
###d5 by day###
all_data = values_normalization(all_data, periods, ['D5'])
###d4 by day###
all_data = values_normalization(all_data, periods, ['D4'])


##new variable hour
all_data['hour_trans_count']=all_data[['hour','TransactionAmt']].groupby(['hour']).transform('count')

temp=all_data[['addr1','hour']].groupby(['addr1','hour'])['hour'].agg(['count'])
temp=temp.reset_index()
temp.sort_values(by=['addr1','count'], inplace=True)
temp=temp.drop_duplicates(subset=['addr1'], keep='last')
temp.index = temp['addr1'].values
temp_dict = temp['hour'].to_dict()
all_data['most_common_hour_addr1']=all_data['addr1'].map(temp_dict)
all_data['most_common_hour_addr1'].head()

all_data['hour_diff']=all_data['hour']-all_data['most_common_hour_addr1']
del temp,temp_dict
gc.collect()




# label encoder
cat_cols=[]
for col in all_data.columns:
    if all_data[col].dtype=='object':
        le=LabelEncoder()
        le.fit(list(all_data[col].astype(str).values))
        all_data[col]=le.transform(list(all_data[col].astype(str).values))
        cat_cols.append(col)
print('Cat Cols:',cat_cols)




###reduce memory again
reduce_cols=list(all_data.columns)
reduce_cols.remove('TransactionAmt')
reduce_cols.remove('TransactionAmt_decimal')
all_data=reduce_mem_usage(all_data)
gc.collect()




#do unique count

all_data = do_countuniq(all_data, ['uid4'], 'TransactionAmt' ); gc.collect()

all_data = do_countuniq(all_data, ['uid4'], 'ProductCD' ); gc.collect()

all_data['C1_PCD']=all_data['C1'].astype(str)+'_'+all_data['ProductCD'].astype(str)
all_data = do_countuniq(all_data, ['C1_PCD'], 'C8' ); gc.collect()
all_data = do_countuniq(all_data, ['C1_PCD'], 'D3' ); gc.collect()

del all_data['C1_PCD']



#split
all_data.drop(cols_to_drop, axis=1,inplace=True)
train=all_data[all_data['isFraud']!=-1]#####split train test#####
test=all_data[all_data['isFraud']==-1]
del all_data
gc.collect()

##process D features
#frequency count by time

i_cols=['TransactionAmt']
periods=['DT_D','DT_W']
train, test = timeblock_frequency_encoding(train, test, periods, i_cols, 
                                                 with_proportions=False, only_proportions=False)
i_cols=['C1']
periods=['uid4']
train, test = timeblock_frequency_encoding(train, test, periods, i_cols, 
                                                 with_proportions=False, only_proportions=False)

###drop some time dependent variables####
cols_to_drop.append('id_31')
new_todrop=['DT','DT_M','DT_W','DT_D','DT_hour','DT_day_week','DT_day_month','uid','uid2','uid4']###uid4 NEW###
train.drop(new_todrop+cols_to_drop,inplace=True,axis=1)
test.drop(new_todrop+cols_to_drop,inplace=True,axis=1)


train.to_csv('train_processed.csv',index=False)
test.to_csv('test_processed.csv',index=False)

