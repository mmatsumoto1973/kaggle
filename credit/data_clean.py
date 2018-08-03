#%%
import pandas as pd
import csv as csv
from sklearn.ensemble import RandomForestClassifier

# Load training data
train_df = pd.read_csv("./input/application_train.csv", header=0)

# 欠損テーブルの情報を確認する
def kesson_table(df): 
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(
    columns = {0 : '欠損数', 1 : '%'})
    return kesson_table_ren_columns
 
kesson_table(train_df) 


#%%
# TARGET との相関関係の低い(絶対値が0.001未満)のcolumnを削除
train_df = train_df.drop(["FLAG_DOCUMENT_2", "FLAG_DOCUMENT_3", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_5", "FLAG_DOCUMENT_6", "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_8", "FLAG_DOCUMENT_9", "FLAG_DOCUMENT_10", "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13", "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16", "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21"], axis=1)
# 欠損率40%以上のcolumnを削除
train_df = train_df.drop(["OWN_CAR_AGE", "APARTMENTS_MEDI", "BASEMENTAREA_MEDI", "YEARS_BEGINEXPLUATATION_MEDI", "YEARS_BUILD_MEDI", "COMMONAREA_MEDI", "ELEVATORS_MEDI", "ENTRANCES_MEDI", "FLOORSMAX_MEDI", "FLOORSMIN_MEDI", "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI", "LIVINGAREA_MEDI", "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAREA_MEDI", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "TOTALAREA_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE"], axis=1)
train_df = train_df.drop(["LANDAREA_AVG", "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAREA_AVG", "APARTMENTS_MODE", "BASEMENTAREA_MODE", "YEARS_BEGINEXPLUATATION_MODE", "YEARS_BUILD_MODE", "COMMONAREA_MODE", "ELEVATORS_MODE", "ENTRANCES_MODE", "FLOORSMAX_MODE", "FLOORSMIN_MODE", "LANDAREA_MODE", "LIVINGAPARTMENTS_MODE", "LIVINGAREA_MODE", "NONLIVINGAPARTMENTS_MODE", "NONLIVINGAREA_MODE"], axis=1)
train_df = train_df.drop(["EXT_SOURCE_1", "APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BUILD_AVG", "COMMONAREA_AVG", "ELEVATORS_AVG", "ENTRANCES_AVG", "FLOORSMAX_AVG", "FLOORSMIN_AVG"], axis=1)
#kesson_table(train_df) 

# AMT_ANNUITY クレジットカードローン
#%%
#pd.value_counts(train_df['AMT_ANNUITY']) 
train_df.fillna({'AMT_ANNUITY': train_df['AMT_ANNUITY'].mean()}, inplace = True)
#kesson_table(train_df) 

# AMT_GOODS_PRICE  ローン利率？
#%%
#train_df['AMT_GOODS_PRICE'].value_counts()
train_df.fillna({'AMT_GOODS_PRICE': train_df['AMT_GOODS_PRICE'].mean()}, inplace = True)
#kesson_table(train_df) 

# NAME_TYPE_SUIT  顧客がローンを組む時に誰が付き添っていたか
#%%
# null値レコードは削除
#train_df['NAME_TYPE_SUITE'].value_counts()
train_df.dropna(subset=['NAME_TYPE_SUITE'], inplace = True)
#kesson_table(train_df) 

# OCCUPATION_TYPE  クライアントの職業
#%%
# null値レコードは削除
#train_df['OCCUPATION_TYPE'].value_counts()
train_df.dropna(subset=['OCCUPATION_TYPE'], inplace = True)
#kesson_table(train_df) 

# EXT_SOURCE_2  クライアントの職業
#%%
# null値レコードは平均値を入れる
train_df.fillna({'EXT_SOURCE_2': train_df['EXT_SOURCE_2'].mean()}, inplace = True)


# EXT_SOURCE_3  クライアントの職業
#%%
# null値レコードは平均値を入れる
train_df.fillna({'EXT_SOURCE_3': train_df['EXT_SOURCE_3'].mean()}, inplace = True)


# OBS_30_CNT_SOCIAL_CIRCLE
#%%
# null値レコードは平均値を入れる
#train_df['OBS_30_CNT_SOCIAL_CIRCLE'].value_counts()
train_df.fillna({'OBS_30_CNT_SOCIAL_CIRCLE': train_df['OBS_30_CNT_SOCIAL_CIRCLE'].mean()}, inplace = True)

# DEF_30_CNT_SOCIAL_CIRCLE
#%%
# null値レコードは平均値を入れる
#train_df['DEF_30_CNT_SOCIAL_CIRCLE'].value_counts()
train_df.fillna({'DEF_30_CNT_SOCIAL_CIRCLE': train_df['DEF_30_CNT_SOCIAL_CIRCLE'].mean()}, inplace = True)

# OBS_60_CNT_SOCIAL_CIRCLE
#%%
# null値レコードは平均値を入れる
#train_df['OBS_60_CNT_SOCIAL_CIRCLE'].value_counts()
train_df.fillna({'OBS_60_CNT_SOCIAL_CIRCLE': train_df['OBS_60_CNT_SOCIAL_CIRCLE'].mean()}, inplace = True)

# DEF_60_CNT_SOCIAL_CIRCLE
#%%
# null値レコードは平均値を入れる
#train_df['DEF_60_CNT_SOCIAL_CIRCLE'].value_counts()
train_df.fillna({'DEF_60_CNT_SOCIAL_CIRCLE': train_df['DEF_60_CNT_SOCIAL_CIRCLE'].mean()}, inplace = True)

# DAYS_LAST_PHONE_CHANGE
#%%
# null値レコードは平均値を入れる
#train_df['DAYS_LAST_PHONE_CHANGE'].value_counts()
train_df.dropna(subset=['DAYS_LAST_PHONE_CHANGE'], inplace = True)