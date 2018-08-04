
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
kesson_table(train_df) 

# AMT_ANNUITY クレジットカードローン
#%%
pd.value_counts(train_df['AMT_ANNUITY']) 
train_df.fillna({'AMT_ANNUITY': train_df['AMT_ANNUITY'].mean()}, inplace = True)
kesson_table(train_df) 

# AMT_GOODS_PRICE  ローン利率？
#%%
train_df['AMT_GOODS_PRICE'].value_counts()
train_df.fillna({'AMT_GOODS_PRICE': train_df['AMT_GOODS_PRICE'].mean()}, inplace = True)
kesson_table(train_df) 

# NAME_TYPE_SUIT  顧客がローンを組む時に誰が付き添っていたか
#%%
# null値レコードは削除
train_df['NAME_TYPE_SUITE'].value_counts()
train_df.dropna(subset=['NAME_TYPE_SUITE'], inplace = True)
kesson_table(train_df) 

# OCCUPATION_TYPE  クライアントの職業
#%%
# null値レコードは削除
train_df['OCCUPATION_TYPE'].value_counts()
train_df.dropna(subset=['OCCUPATION_TYPE'], inplace = True)
kesson_table(train_df) 

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
train_df['OBS_30_CNT_SOCIAL_CIRCLE'].value_counts()
train_df.fillna({'OBS_30_CNT_SOCIAL_CIRCLE': train_df['OBS_30_CNT_SOCIAL_CIRCLE'].mean()}, inplace = True)

# DEF_30_CNT_SOCIAL_CIRCLE
#%%
# null値レコードは平均値を入れる
train_df['DEF_30_CNT_SOCIAL_CIRCLE'].value_counts()
train_df.fillna({'DEF_30_CNT_SOCIAL_CIRCLE': train_df['DEF_30_CNT_SOCIAL_CIRCLE'].mean()}, inplace = True)

# OBS_60_CNT_SOCIAL_CIRCLE
#%%
# null値レコードは平均値を入れる
train_df['OBS_60_CNT_SOCIAL_CIRCLE'].value_counts()
train_df.fillna({'OBS_60_CNT_SOCIAL_CIRCLE': train_df['OBS_60_CNT_SOCIAL_CIRCLE'].mean()}, inplace = True)

# DEF_60_CNT_SOCIAL_CIRCLE
#%%
# null値レコードは平均値を入れる
train_df['DEF_60_CNT_SOCIAL_CIRCLE'].value_counts()
train_df.fillna({'DEF_60_CNT_SOCIAL_CIRCLE': train_df['DEF_60_CNT_SOCIAL_CIRCLE'].mean()}, inplace = True)

# DAYS_LAST_PHONE_CHANGE
#%%
# null値レコードは平均値を入れる
train_df['DAYS_LAST_PHONE_CHANGE'].value_counts()
train_df.dropna(subset=['DAYS_LAST_PHONE_CHANGE'], inplace = True)

#%%
df_corr = train_df.corr()
print(df_corr)
#train_df.fillna({'NAME_TYPE_SUITE': train_df['NAME_TYPE_SUITE'].mean()}, inplace = True)





#%%
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
train_df['NAME_TYPE_SUITE'] = le.fit_transform(train_df['NAME_TYPE_SUITE'])
train_df['NAME_TYPE_SUITE'].value_counts()
df_corr = train_df.corr()
display(df_corr)

#%%
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

#%%
# Convert "Sex" to be a dummy variable (female = 0, Male = 1)
train_df["Gender"] = train_df["Sex"].map({"female": 0, "male": 1}).astype(int)
train_df.head(3)

# Complement the missing values of "Age" column with average of "Age"
median_age = train_df["Age"].dropna().median()
if len(train_df.Age[train_df.Age.isnull()]) > 0:
  train_df.loc[(train_df.Age.isnull()), "Age"] = median_age

# remove un-used columns
train_df = train_df.drop(["Name", "Ticket", "Sex", "SibSp", "Parch", "Fare", "Cabin", "Embarked", "PassengerId"], axis=1)
train_df.head(3)

# Load test data, Convert "Sex" to be a dummy variable
test_df = pd.read_csv("test.csv", header=0)
test_df["Gender"] = test_df["Sex"].map({"female": 0, "male": 1}).astype(int)

# Complement the missing values of "Age" column with average of "Age"
median_age = test_df["Age"].dropna().median()
if len(test_df.Age[test_df.Age.isnull()]) > 0:
  test_df.loc[(test_df.Age.isnull()), "Age"] = median_age

# Copy test data's "PassengerId" column, and remove un-used columns
ids = test_df["PassengerId"].values
test_df = test_df.drop(["Name", "Ticket", "Sex", "SibSp", "Parch", "Fare", "Cabin", "Embarked", "PassengerId"], axis=1)
test_df.head(3)

# Predict with "Random Forest"
train_data = train_df.values
test_data = test_df.values
model = RandomForestClassifier(n_estimators=100)
output = model.fit(train_data[0::, 1::], train_data[0::, 0]).predict(test_data).astype(int)

# export result to be "card_submit.csv"
submit_file = open("./output/card_submit.csv", "w")
file_object = csv.writer(submit_file)
file_object.writerow(["SK_ID_CURR", "TARGET"])
file_object.writerows(zip(ids, output))
submit_file.close()