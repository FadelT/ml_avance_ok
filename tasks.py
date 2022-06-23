
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_iris,load_wine, load_diabetes
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

def generer_base_donnee(nature,nb_row,nb_cols,nb_class):
    if nature=='regression':
        data=np.random.randn(nb_row,nb_cols)
        print('ok')
        print(data)
        
    elif nature=='classification':
        if nb_class==None:
            st.write('Veuillez spécifiez le nombre de classe voulu')
        else:
            data=make_classification(n_samples=nb_row,n_features=nb_cols, n_classes=nb_class)
    return data

def use_sklearn_datasets(donnee):
    if donnee=='iris':
        data=load_iris()
    elif donnee=='wine':
        data=load_wine()
    elif donnee=='diabetes':
        data=load_diabetes(return_X_y=True,as_frame=True)
    else:
        print('ok')
    return data

def caracteristics_data(data_frame):
    X_columns=data_frame.columns[:-1]
    y_columns=data_frame.columns[-1]
    nb_rows=data_frame.shape[0]
    nb_cols=data_frame.shape[1]
    descri=data_frame.describe()
    return descri

def choix_donnee(attributs,nb_rows,data):
    data=data.loc[:nb_rows,attributs]
    return data


def normaliser(X_data,technique):
    if technique=='standardScaler':
        scaler = StandardScaler()
    elif technique=='MinMax':
        scaler = MinMaxScaler()
    elif technique=='Normal':
        scaler = Normalizer()

    scaler.fit(X_data)
    X_data_transformed=scaler.transform(X_data)
    return X_data_transformed

def entrain_algo(nature_task,X,y,test_size,parametres_model):
    print(X)
    print(y.shape)
    if nature_task=='classification':
        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.15,stratify=y)
    elif nature_task=='regression':
        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.15)

    if nature_task=='classification':
        if parametres_model['name']=='LogisticRegression':
            model=LogisticRegression(penalty=parametres_model['penalty'],solver=parametres_model['solver'])

        elif parametres_model['name']=='KNeighborclassifier':  
            model=KNeighborsClassifier(n_neighbors=parametres_model['nb_neighbors'])
        
        elif  parametres_model['name']=='XgboostClassifier':
            model=XGBClassifier(n_estimators=parametres_model['n_estimators'], max_depth=parametres_model['max_depth'], eta=parametres_model['eta'], subsample=parametres_model['subsample'], colsample_bytree=parametres_model['colsample_bytree'])

    elif nature_task=='regression':
        if parametres_model['name']=='LinearRegression':
            model=LinearRegression()

        elif parametres_model['name']=='KNeighborRegressor':  
            model=KNeighborsRegressor(n_neighbors=parametres_model['nb_neighbors'])
        
        elif  parametres_model['name']=='XgboostRegressor':

            model=XGBRegressor(n_estimators=parametres_model['n_estimators'], max_depth=parametres_model['max_depth'], eta=parametres_model['eta'], subsample=parametres_model['subsample'], colsample_bytree=parametres_model['colsample_bytree'])
    

    model.fit(X_train, y_train)
    if parametres_model['name']=='XgboostClassifier' or parametres_model['name']=='XgboostRegressor':
       y_pred= model.predict(X_test)
       if parametres_model['name']=='XgboostClassifier':
            y_pred = [round(value) for value in y_pred]
       
    else:
         y_pred=model.predict(X_test)
    if nature_task=='classification':
        score=accuracy_score(y_test,y_pred)
    else:
        score=mean_squared_error(y_test,y_pred)

    return score

    
