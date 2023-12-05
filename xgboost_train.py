import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error


xgb.set_config(verbosity=0)

test=pd.read_csv('cleaned_data-test.csv',low_memory=False)
#print('test_dataset：\n',test.shape)
vali=pd.read_csv('cleaned_data-validate.csv',low_memory=False)
#print('validate_dataset：\n',vali.shape)
train=pd.read_csv('cleaned_data-train.csv',low_memory=False)
#print('train_dataset：\n',train.shape)
print(pd.isnull(train).values.any()) #查看损失值

ntrain=train.shape[0]

features1=[x for x in train.columns
          if x not in ['overall_span']]
train_x=train[features1]
train_y=train['overall_span']
print('Xtrain:',train_x.shape)
print('ytrain:',train_y.shape)
dtrain=xgb.DMatrix(train_x,train['overall_span'])

ntest=test.shape[0]
features2=[x for x in test.columns
          if x not in ['overall_span']]
test_x=test[features2]
test_y=test['overall_span']
# 2.参数集定义
param_grid = {
    'max_depth': [2, 3],
    'n_estimators': [30, 50],
    'learning_rate': [0.1, 0.2],
    "gamma": [0.0, 0.1],
    "reg_alpha": [0.0001, 0.001],
    "reg_lambda": [0.0001, 0.001],
    "min_child_weight": [2, 3],
    "colsample_bytree": [0.6, 0.7],
    "subsample": [ 0.8, 0.9]}
# 3.随机搜索并打印最佳参数
gsearch1 = RandomizedSearchCV(XGBRegressor(scoring='ls', seed=27), param_grid, cv=5)
gsearch1.fit(train_x, train_y)
print("best_score_:", gsearch1.best_params_, gsearch1.best_score_)

# 4.用最佳参数进行预测
y_test_pre = gsearch1.predict(test_x)

# 5.打印测试集RMSE
rmse = sqrt(mean_squared_error(np.array(list(test_y)), np.array(list(y_test_pre))))
print("rmse:", rmse)



