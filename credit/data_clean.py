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

# データ処理前の相関関係を確認
#%%
# TARGET との相関関係の低い(絶対値が0.001未満)のcolumnを削除
train_df = train_df.drop(["FLAG_DOCUMENT_2", "FLAG_DOCUMENT_3", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_5", "FLAG_DOCUMENT_6", "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_8", "FLAG_DOCUMENT_9", "FLAG_DOCUMENT_10", "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13", "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16", "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21"], axis=1)
train_df = train_df.drop(["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK"], axis=1)
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
# null値レコードは削除
#train_df['DAYS_LAST_PHONE_CHANGE'].value_counts()
train_df.dropna(subset=['DAYS_LAST_PHONE_CHANGE'], inplace = True)

# AMT_REQ_CREDIT_BUREAU_HOUR
#%%
# null値レコードは平均値を入れる
train_df.fillna({'AMT_REQ_CREDIT_BUREAU_HOUR': train_df['AMT_REQ_CREDIT_BUREAU_HOUR'].mean()}, inplace = True)

# AMT_REQ_CREDIT_BUREAU_DAY
#%%
# null値レコードは平均値を入れる
train_df.fillna({'AMT_REQ_CREDIT_BUREAU_DAY': train_df['AMT_REQ_CREDIT_BUREAU_DAY'].mean()}, inplace = True)

# AMT_REQ_CREDIT_BUREAU_WEEK
#%%
# null値レコードは平均値を入れる
train_df.fillna({'AMT_REQ_CREDIT_BUREAU_WEEK': train_df['AMT_REQ_CREDIT_BUREAU_WEEK'].mean()}, inplace = True)

# AMT_REQ_CREDIT_BUREAU_MON
#%%
# null値レコードは平均値を入れる
train_df.fillna({'AMT_REQ_CREDIT_BUREAU_MON': train_df['AMT_REQ_CREDIT_BUREAU_MON'].mean()}, inplace = True)

# AMT_REQ_CREDIT_BUREAU_QRT
#%%
# null値レコードは平均値を入れる
train_df.fillna({'AMT_REQ_CREDIT_BUREAU_QRT': train_df['AMT_REQ_CREDIT_BUREAU_QRT'].mean()}, inplace = True)

# AMT_REQ_CREDIT_BUREAU_YEAR
#%%
# null値レコードは平均値を入れる
train_df.fillna({'AMT_REQ_CREDIT_BUREAU_YEAR': train_df['AMT_REQ_CREDIT_BUREAU_YEAR'].mean()}, inplace = True)

# 以降はカラム間の相関関係を出力するための処理のため、取り込むかどうかの判断が必要

# 文字列から数値に変換
#%%
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

train_df['NAME_CONTRACT_TYPE'] = le.fit_transform(train_df['NAME_CONTRACT_TYPE'])
train_df['NAME_CONTRACT_TYPE'].value_counts()
train_df['CODE_GENDER'] = le.fit_transform(train_df['CODE_GENDER'])
train_df['CODE_GENDER'].value_counts()
train_df['FLAG_OWN_CAR'] = le.fit_transform(train_df['FLAG_OWN_CAR'])
train_df['FLAG_OWN_CAR'].value_counts()
train_df['FLAG_OWN_REALTY'] = le.fit_transform(train_df['FLAG_OWN_REALTY'])
train_df['FLAG_OWN_REALTY'].value_counts()
train_df['NAME_TYPE_SUITE'] = le.fit_transform(train_df['NAME_TYPE_SUITE'])
train_df['NAME_TYPE_SUITE'].value_counts()
train_df['NAME_INCOME_TYPE'] = le.fit_transform(train_df['NAME_INCOME_TYPE'])
train_df['NAME_INCOME_TYPE'].value_counts()
train_df['NAME_EDUCATION_TYPE'] = le.fit_transform(train_df['NAME_EDUCATION_TYPE'])
train_df['NAME_EDUCATION_TYPE'].value_counts()
train_df['NAME_FAMILY_STATUS'] = le.fit_transform(train_df['NAME_FAMILY_STATUS'])
train_df['NAME_FAMILY_STATUS'].value_counts()
train_df['NAME_HOUSING_TYPE'] = le.fit_transform(train_df['NAME_HOUSING_TYPE'])
train_df['NAME_HOUSING_TYPE'].value_counts()
train_df['OCCUPATION_TYPE'] = le.fit_transform(train_df['OCCUPATION_TYPE'])
train_df['OCCUPATION_TYPE'].value_counts()
train_df['WEEKDAY_APPR_PROCESS_START'] = le.fit_transform(train_df['WEEKDAY_APPR_PROCESS_START'])
train_df['WEEKDAY_APPR_PROCESS_START'].value_counts()
train_df['ORGANIZATION_TYPE'] = le.fit_transform(train_df['ORGANIZATION_TYPE'])
train_df['ORGANIZATION_TYPE'].value_counts()
kesson_table(train_df) 

#%%
# カラム間の相関関係を出力する 　
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df_corr = train_df.corr()
display(df_corr)

#%%
# ヒートマップの出力
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df_corr,
            vmin=-1.0,
            vmax=1.0,
            center=0,
            annot=True, # True:格子の中に値を表示
            fmt='.1f',
            xticklabels=df_corr.columns.values,
            yticklabels=df_corr.columns.values
           )
plt.show()