import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor,RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split,GridSearchCV
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import joblib

bag=BaggingRegressor()
rf=RandomForestRegressor()
svr=SVR()
lgbm=LGBMRegressor()

model_list=[bag,rf,svr]

df=pd.read_csv(r"C:\Users\safak\spyder projects\clothes\safak_erkek_kazak.csv")
#df.drop([214],axis=0,inplace=True)
encoding=pd.get_dummies(df)
df=encoding.copy()
X=df.drop(columns=["Key","Like"])
y=df["Like"]


def model_training(model_list,X,y,test_size=0.2):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=1)
    
    mae_dict=dict()
    
    #Train
    for model in model_list:
        
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        
        
        mae_dict[(f"{model}",model)]=mean_absolute_error(y_pred,y_test)
    
    return [mae_dict,[X_train,X_test,y_train,y_test]]

def model_performance_plotting(models,scores):
    
    
    plt.bar(models,scores)
    plt.xlabel("Model Names")
    plt.ylabel("MAE scores")
    plt.title("Model comparison by MAE")
    plt.xticks(rotation=60)
    plt.show() 


mae_dict,datasets=model_training(model_list,X,y,test_size=0.2)
mae_dict
models=[i[0] for i in list(mae_dict.keys())]
scores=list(mae_dict.values())
model_performance_plotting(models,scores)

def model_evaluating(mae_dict,X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    key_list=list(mae_dict.keys())
    val_list=list(mae_dict.values())
    pos=val_list.index(min(val_list))
    best_model=key_list[pos]
    before_best_model_score=min(val_list)
    
    
    """if best_model=="xgboost":
        param_grid={"learning_rate": [0.01,0.1],
                   "max_depth":[3,5,7],
                   "n_estimators":[100,200,500],
                   "objective":["reg:absoluteerror"]}
        est=XGBRegressor()
        grid_search=GridSearchCV(estimator=est,param_grid=param_grid,cv=3)
        grid_search.fit(X_train,y_train)
        best_estimator=grid_search.best_estimator_
        best_estimator.fit(X_train,y_train)
        new_y_pred=best_estimator.predict(X_test)
        new_mae=mean_absolute_error(new_y_pred,y_test)
        if(before_best_model_score>new_mae):
            print("kontrol 1")
            return best_estimator,new_mae
        else:
            print("kontrol 2")
            xgb=XGBRegressor()
            xgb.fit(X_train,y_train)
            y_pred=xgb.predict(X_test)
            mae=mean_absolute_error(y_pred,y_test)
            return xgb,mae"""
    
    
    if best_model[0]=="RandomForestRegressor()":
        
        
        param_grid={
            
            "max_depth":[80,90,100,110],
            "min_samples_leaf":[3,4,5],
            "min_samples_split":[8,10,12],
            "n_estimators":[100,200,300,1000]}
        
        est=RandomForestRegressor()
        grid_search=GridSearchCV(estimator=est,param_grid=param_grid,cv=3)
        grid_search.fit(X_train,y_train)
        best_estimator=grid_search.best_estimator_
        best_estimator.fit(X_train,y_train)
        new_y_pred=best_estimator.predict(X_test)
        new_mae=mean_absolute_error(new_y_pred,y_test)
        if(before_best_model_score>new_mae):
            
            return best_estimator,new_mae
        else:
            
            return best_model[1],mae_dict[best_model]
    
    if best_model[0]=="BaggingRegressor()":
        
        param_grid={"n_estimators":range(2,20)}
        est=BaggingRegressor()
        grid_search=GridSearchCV(estimator=est,param_grid=param_grid,cv=3)
        grid_search.fit(X_train,y_train)
        best_estimator=grid_search.best_estimator_
        best_estimator.fit(X_train,y_train)
        new_y_pred=best_estimator.predict(X_test)
        new_mae=mean_absolute_error(new_y_pred,y_test)
        if(before_best_model_score>new_mae):
            print("kontrol 1")
            return best_estimator,new_mae
        
        else:
            
            
            return best_model[1],mae_dict[best_model]
        
    
    if best_model[0]=="SVR()":
        
        param_grid={"C":[0.1,0.2,0.4,0.8,3,4,5,10,15,20,25,30,40,50]}
        est=SVR()
        grid_search=GridSearchCV(estimator=est,param_grid=param_grid,cv=3)
        grid_search.fit(X_train,y_train)
        best_estimator=grid_search.best_estimator_
        best_estimator.fit(X_train,y_train)
        new_y_pred=best_estimator.predict(X_test)
        new_mae=mean_absolute_error(new_y_pred,y_test)
        if(before_best_model_score>new_mae):
            
            return best_estimator,new_mae
        
        else:
            
            return best_model[1],mae_dict[best_model]
        
    
    if best_model[0]=="LGBMRegressor()":
        
        
        return best_model[1],mae_dict[best_model]
best_model,best_score=model_evaluating(mae_dict,X,y)

best_model

best_score

#joblib.dump(best_model,"./MODEL_erkek_kazak.joblib")

#--------predict-----------

"""mom=joblib.load("MODEL_erkek_tshirt.joblib")

xtest=datasets[1]
ytest=datasets[3]

tahmin=df.iloc[117]
tahmin=tahmin.drop(["Key","Like"])
best_model.predict([list(tahmin)])
mom.predict([list(tahmin)])"""

#*******************************************************

"""
erkek_sweat -> lgbm (0.11) -> [bagging (0.11)]
erkek sort -> svr (0.10) -> [svr (0.10)]
erkek tshirt -> bagging (0.089) -> [bagging (0.085)]
kadin tshirt -> svr (0.094) -> [svr (0.091)]○
kadin abiye -> knn (0.097) -> [knn (0.10)]
kadin sort -> rf (0.097) -> [svr (0.10)]
"""
"""
erkek gomlek -> rf (0.083)
erkek kazak -> bagging (0.10)
erkek pantolon -> svr (0.082)
kadin etek -> rf (0.10)
kadin gomlek -> rf (0.089)
kadin kazak -> svr (0.094)
kadin pantolon -> bagging (0.10)
"""












