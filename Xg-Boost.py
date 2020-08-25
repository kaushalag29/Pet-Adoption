#!/usr/bin/env python3
# Import Required Modules
import datetime
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from xgboost import XGBClassifier
#%% Loading Train Data & Shuffle It
train_df = pd.read_csv("train.csv", na_values = '?', parse_dates = True).sample(frac = 1, random_state = 0)
#%% Drop Un-necessary Rows
train_df.drop(train_df[(train_df["color_type"] == "Brown Tiger")].index, axis = 0, inplace = True)
train_df.drop(train_df[(train_df["color_type"] == "Black Tiger")].index, axis = 0, inplace = True)
#%% Load Test Data
test_df = pd.read_csv("test.csv", na_values = '?', parse_dates = True)
#%% Resetting Index Since Training Data Was Shuffled
train_df.reset_index(drop = True, inplace = True)
#%% Cecking Columns Containg Missing Values
print(train_df.isna().sum())
print("-----------------------------------")
print(test_df.isna().sum())
print("-----------------------------------")
#%% Splitting Date And Time
def getDateAndTime(data):
    dateAndTime = data.split(' ')
    date = dateAndTime[0].split('-')
    int_date = [int(i) for i in date]
    time = dateAndTime[1].split(':')
    int_time = [int(i) for i in time]
    return (int_date, int_time)
#%% Taking Difference Between Listing & Issue Date To Make A Useful Feature
def dateAndTimeDiff(issue_df, list_df):
    diff_df1 = []
    diff_df2 = []
    idx = 0
    for data in issue_df:
        x = getDateAndTime(data)
        y = getDateAndTime(list_df[idx])
        idx += 1
        time1 = datetime.datetime(x[0][0], x[0][1], x[0][2],
                                  x[1][0], x[1][1], x[1][2])
        time2 = datetime.datetime(y[0][0], y[0][1], y[0][2],
                                  y[1][0], y[1][1], y[1][2])
        diff1 = (time2-time1).days
        diff2 = (time2-time1).seconds
        diff_df1.append(diff1)
        diff_df2.append(diff2)
    return (pd.DataFrame(diff_df1), pd.DataFrame(diff_df2))
#%% Adding Columns To Training Set
diff_dfs = dateAndTimeDiff(train_df["issue_date"], train_df["listing_date"])
train_df["diff1"] = diff_dfs[0]
train_df["diff2"] = diff_dfs[1] 
#%% Adding Colums to Test Set
diff_dfs = dateAndTimeDiff(test_df["issue_date"], test_df["listing_date"])
test_df["diff1"] = diff_dfs[0]
test_df["diff2"] = diff_dfs[1]
#%% Initiating A Solution Dataframe
solution_df = test_df["pet_id"].to_frame()
#%% Drop Un-necessary Attributes From Training & Test Set
def dropUnnecessaryAttr(df):
    df.drop(["pet_id", "issue_date", "listing_date"], axis = 1, inplace = True)
    return df
#%%
train_df = dropUnnecessaryAttr(train_df)
test_df = dropUnnecessaryAttr(test_df)
#%%
train_Y_df = train_df.loc[:, ["breed_category"]]
train_X_df = train_df.drop(["breed_category", "pet_category"], axis = 1)
#%% Initialize MinMaxScaler For Feature Scaling
min_max_scaler = preprocessing.MinMaxScaler()
#%% Normalize Test Set
test_df["height(cm)"] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(test_df["height(cm)"])))
test_df["length(m)"] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(test_df["length(m)"])))
test_df["diff1"] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(test_df["diff1"])))
test_df["diff2"] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(test_df["diff2"])))
#%% Normalize Train Set
train_X_df["diff1"] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(train_X_df["diff1"])))
train_X_df["diff2"] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(train_X_df["diff2"])))
train_X_df["height(cm)"] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(train_X_df["height(cm)"])))
train_X_df["length(m)"] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(train_X_df["length(m)"])))
#%% Convert Categorical Values To Numerical Values
def encodeCategoricalColumn(df):
    return pd.get_dummies(df["color_type"], drop_first = True)
#%% For Training Set
get_encoded_colors = encodeCategoricalColumn(train_X_df)
train_X_df.drop(["color_type"], axis = 1, inplace = True)
train_X_df = pd.concat([train_X_df, get_encoded_colors], axis = 1)
#%% For Test Set
get_encoded_colors = encodeCategoricalColumn(test_df)
test_df.drop(["color_type"], axis = 1, inplace = True)
test_df = pd.concat([test_df, get_encoded_colors], axis = 1)
#%% Add Extra Column For Missing Data --> 1 Else 0 For 'condition' Column
test_df["condition_ismissing"] = pd.isnull(test_df["condition"]).astype(np.int)
train_X_df["condition_ismissing"] = pd.isnull(train_X_df["condition"]).astype(np.int)
#%% Filling Missing Values In Train & Test Set With Imputer
imputer = KNNImputer(n_neighbors = 200)
imputer.fit(train_X_df)
train_X_df = imputer.transform(train_X_df)
train_X_df = pd.DataFrame(train_X_df)
imputer = KNNImputer(n_neighbors = 100)
imputer.fit(test_df)
test_df = imputer.transform(test_df)
test_df = pd.DataFrame(test_df)
#%% XgBoost Classifier Initialized With Hyper-Tuned Params
model = XGBClassifier(n_estimators = 1000,
                      learning_rate = 0.05,
                      random_state = 1,
                      max_depth = 3,
                      gamma = 1,
                      max_delta_step = 0,
                      min_child_weight = 3,
                      reg_lambda = 1,
                      reg_alpha = 0,
                      colsample_bytree = 0.55,
                      subsample = 1,
                      scale_pos_weight = 1,
                      base_score = 0.5,
                      colsample_bylevel = 1,
                      colsample_bynode = 1,
                      objective = "multi:softmax",
                      gpu_id = 0)
#%% Fitting The Model
%timeit -n1 -r1 model.fit(train_X_df, train_Y_df.values.ravel())
#%% Making Predictions
prediciton = model.predict(test_df)
pred_df = pd.DataFrame(prediciton)
pred_df.columns = ["breed_category"]
pred_df = pred_df.astype(int)
#%% Updating Solution Dataframe
solution_df = pd.concat([solution_df, pred_df], axis = 1)
#%% Changing Target Column
train_Y_df = pd.DataFrame(train_df["pet_category"])
#%% Adding Predicted Value To Test Set & Given Value To Training Set
test_df["breed_category"] = pred_df["breed_category"]
train_X_df["breed_category"] = train_df["breed_category"]
#%% Again Fit The Model
print("-----------------------------------")
%timeit -n1 -r1 model.fit(train_X_df, train_Y_df.values.ravel())
print("-----------------------------------")
#%% Make Predictions
prediciton = model.predict(test_df)
pred_df = pd.DataFrame(prediciton)
pred_df.columns = ["pet_category"]
pred_df = pred_df.astype(int)
#%% Update Solution Dataframe & Write To CSV
solution_df = pd.concat([solution_df, pred_df], axis = 1)
solution_df.to_csv("Submission.csv", index = False)
