# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 18:51:22 2024

@author: safak
"""

from sklearn.neural_network import MLPRegressor
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split,ShuffleSplit,GridSearchCV,KFold, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error, mean_squared_log_error
import joblib
df=pd.read_csv(r"C:\Users\safak\spyder projects\clothes\safak_erkek_kazak.csv")
#df.drop([214],axis=0,inplace=True)
#df=pd.read_csv("pout.csv")
df_1=df.copy()
df_2=df_1.copy()
#df_1.drop(["Ortam_Party"],inplace=True,axis=1)

#df_1=df_2.iloc[:,:9]

y=df_1["Like"]
x=df_1.drop(["Like"],axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=10)

key_train=x_train["Key"]
key_test=x_test["Key"]


x_train.drop(["Key"],axis=1,inplace=True)
x_test.drop(["Key"],axis=1,inplace=True)
#----------------------------------
result={}
#----------------------------------
temp_lst=[]

mlp_model=MLPRegressor().fit(x_train,y_train)

mlp_params={"alpha":[0.1,0.01,0.02,0.4,0.04,0.004,0.6,0.06,0.006,0.001,0.03,0.2,0.002,0.003,0.005],
            "activation":["relu","logistic","identity","tanh"]}

mlp_cv_model=GridSearchCV(mlp_model,mlp_params,cv=16)
mlp_cv_model.fit(x_train,y_train)

mlp_tuned=MLPRegressor(alpha=mlp_cv_model.best_params_["alpha"],activation=mlp_cv_model.best_params_["activation"])
mlp_tuned.fit(x_train,y_train)

y_pred=mlp_tuned.predict(x_test)

"""print("kök ort kare hatası: {}".format(mean_squared_error(y_test,y_pred)))
print("mutlak ortalama hata: {}".format(mean_absolute_error(y_test,y_pred)))
print("r2 skoru: {}".format(r2_score(y_test, y_pred)))"""

temp_lst.append(mean_squared_error(y_test,y_pred))
temp_lst.append(mean_absolute_error(y_test,y_pred))
temp_lst.append(r2_score(y_test, y_pred))

result["MLP"]=temp_lst

#------------------------------------
temp_lst=[]

from sklearn.svm import SVR
svr_model=SVR(kernel="rbf").fit(x_train,y_train)

svr_params={"C":[0.1,0.2,0.4,0.8,3,4,5,10,15,20,25,30,40,50]}
svr_cv_model=GridSearchCV(svr_model, svr_params,cv=10)
svr_cv_model.fit(x_train,y_train)

svr_tuned=SVR(kernel="rbf",C=pd.Series(svr_cv_model.best_params_)[0]).fit(x_train,y_train)

y_pred=svr_tuned.predict(x_test)

"""print("kök ort kare hatası: {}".format(mean_squared_error(y_test,y_pred)))
print("mutlak ortalama hata: {}".format(mean_absolute_error(y_test,y_pred)))
print("r2 skoru: {}".format(r2_score(y_test, y_pred)))"""

temp_lst.append(mean_squared_error(y_test,y_pred))
temp_lst.append(mean_absolute_error(y_test,y_pred))
temp_lst.append(r2_score(y_test, y_pred))

result["SVR"]=temp_lst
#joblib.dump(svr_tuned,"./MODEL_kadin_sort.joblib")

#----------------------------------
"""temp_lst=[]

from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['l1','l2'],
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 8,
    "num_leaves": 128,  
    "max_bin": 512,
    "num_iterations": 100000
}

gbm = lgb.LGBMRegressor(**hyper_params)

gbm.fit(x_train, y_train,
        eval_set=[(x_test, y_test)],
        eval_metric='l1')

y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration_)"""

"""print("kök ort kare hatası: {}".format(mean_squared_error(y_test,y_pred)))
print("mutlak ortalama hata: {}".format(mean_absolute_error(y_test,y_pred)))
print("r2 skoru: {}".format(r2_score(y_test, y_pred)))"""

"""temp_lst.append(mean_squared_error(y_test,y_pred))
temp_lst.append(mean_absolute_error(y_test,y_pred))
temp_lst.append(r2_score(y_test, y_pred))

result["LGBM 1"]=temp_lst"""

#---------------------------------
temp_lst=[]

from lightgbm import LGBMRegressor
import lightgbm as lgb

model = lgb.LGBMRegressor()
model.fit(x_train, y_train)

expected_y  = y_test
predicted_y = model.predict(x_test)

"""print("kök ort kare hatası: {}".format(mean_squared_error(y_test,y_pred)))
print("mutlak ortalama hata: {}".format(mean_absolute_error(y_test,y_pred)))
print("r2 skoru: {}".format(r2_score(y_test, y_pred)))"""

temp_lst.append(mean_squared_error(y_test,y_pred))
temp_lst.append(mean_absolute_error(y_test,y_pred))
temp_lst.append(r2_score(y_test, y_pred))

result["LGBM 2"]=temp_lst


#----------------------------------
temp_lst=[]

from sklearn.ensemble import BaggingRegressor
bag_model=BaggingRegressor(bootstrap_features=True)
bag_model.fit(x_train,y_train)

params={"n_estimators":range(2,20)}
bag_cv_model=GridSearchCV(bag_model, params, cv=10)
bag_cv_model.fit(x_train,y_train)

bag_tuned=BaggingRegressor(n_estimators=bag_cv_model.best_params_["n_estimators"], random_state=45)
bag_tuned.fit(x_train, y_train)

y_pred=bag_tuned.predict(x_test)

"""print("kök ort kare hatası: {}".format(mean_squared_error(y_test,y_pred)))
print("mutlak ortalama hata: {}".format(mean_absolute_error(y_test,y_pred)))
print("r2 skoru: {}".format(r2_score(y_test, y_pred)))"""

temp_lst.append(mean_squared_error(y_test,y_pred))
temp_lst.append(mean_absolute_error(y_test,y_pred))
temp_lst.append(r2_score(y_test, y_pred))

result["BAGGING TREE"]=temp_lst

"""import joblib
joblib.dump(bag_tuned,"./MODEL_erkek_tshirt.joblib")"""

"""tm=list(df.iloc[80])
tm=tm[1:]
deger=bag_tuned.predict(np.array([tm]))
print(deger)"""

#-----------------------------------
temp_lst=[]

from sklearn.neighbors import KNeighborsRegressor

knn_params={"n_neighbors":np.arange(1,100,1)}
knn_model=KNeighborsRegressor()
knn_cv_model=GridSearchCV(knn_model, knn_params,cv=10)
knn_cv_model.fit(x_train,y_train)

knn_tuned=KNeighborsRegressor(n_neighbors=knn_cv_model.best_params_["n_neighbors"])
knn_tuned.fit(x_train,y_train)

y_pred=knn_tuned.predict(x_test)

"""print("kök ort kare hatası: {}".format(mean_squared_error(y_test,y_pred)))
print("mutlak ortalama hata: {}".format(mean_absolute_error(y_test,y_pred)))
print("r2 skoru: {}".format(r2_score(y_test, y_pred)))"""

temp_lst.append(mean_squared_error(y_test,y_pred))
temp_lst.append(mean_absolute_error(y_test,y_pred))
temp_lst.append(r2_score(y_test, y_pred))

result["KNN"]=temp_lst

#joblib.dump(knn_tuned,"./MODEL_kadin_abiye.joblib")

result_df=pd.DataFrame(result)
result_df.index=["MSE","MAE","R2"]
s=result_df.T

#------------------------------------------
"""cv = KFold(n_splits = 5, random_state=42, shuffle=True)
scores = cross_val_score(model, x_train, y_train, cv = cv)

print(np.mean(scores))


cv2=RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
scores2 = cross_val_score(model, x_train, y_train, cv = cv2)

print(np.mean(scores2))

result_df=pd.DataFrame(result)
result_df.index=["MSE","MAE","R2"]
s=result_df.T"""

#****************************************************
"""from sklearn.ensemble import RandomForestRegressor,BaggingRegressor
import xgboost as xgb
from xgboost import XGBRegressor

#///////////////////////////////////////////////////
bag_model=BaggingRegressor(bootstrap_features=True)
bag_model.fit(x_train,y_train)

params={"n_estimators":range(2,20)}
bag_cv_model=GridSearchCV(bag_model, params, cv=10)
bag_cv_model.fit(x_train,y_train)

bag_tuned=BaggingRegressor(n_estimators=bag_cv_model.best_params_["n_estimators"], random_state=45)
bag_tuned.fit(x_train, y_train)
#///////////////////////////////////////////////////
rf = RandomForestRegressor(random_state=42)
rf.fit(x_train,y_train)

rf_params={"max_depth":list(range(1,10)),
           "max_features":[3,5,10,15,50],
           "n_estimators":[100,200,500,800,1000,2000]}
rf_cv=GridSearchCV(rf, rf_params,cv=10,n_jobs=-1)
rf_cv.fit(x_train,y_train)

rf_tuned=RandomForestRegressor(max_depth=rf_cv.best_params_["max_depth"],
                               max_features=rf_cv.best_params_["max_features"],
                               n_estimators=rf_cv.best_params["n_estimators"])
rf_tuned.fit(x_train,y_train)
#///////////////////////////////////////////////////
xgb_train=xgb.DMatrix(data=x_train,label=y_train)
xgb_test=xgb.DMatrix(data=x_test,label=y_test)

xgb=XGBRegressor()
xgb_params={"colsample_bytree":[0.4,0.5,0.6,0.9,1],
            "n_estimators":[100,200,500,800,1000,2000],
            "max_depth":[2,3,4,5,6],
            "learning_rate":[0.1,0.01,0.5,0.7]}

xgb_cv=GridSearchCV(xgb, param_grid=xgb_params,cv=10,n_jobs=-1,verbose=2)
xgb_cv.fit(x_train,y_train)

xgb_tuned=XGBRegressor(colsample_bytree=xgb_cv.best_params_["colsample_bytree"],
                      learning_rate=xgb_cv.best_params_["learning_rate"],
                      max_depth=xgb_cv.best_params_["max_depth"],
                      n_estimators=xgb_cv.best_params_["n_estimators"])

xgb_tuned.fit(x_train,y_train)"""














