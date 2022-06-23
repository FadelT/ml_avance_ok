from array import array
from email.policy import default
from xml.parsers.expat import model
import streamlit as st
from tasks import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def start_job() :  
    # DESIGN implement changes to the standard streamlit UI/UX
    #st.set_page_config(page_title="streamlit_audio_recorder")
    # Design move app further up and remove top padding
    st.markdown('''<style>.css-1egvi7u {margin-top: -3rem;}</style>''',
        unsafe_allow_html=True)
    st.title('Fast Machine Learning')

    # Design change st.Audio to fixed height of 45 pixels
    st.markdown('''<style>.stAudio {height: 45px;}</style>''',
        unsafe_allow_html=True)
    # Design change hyperlink href link color
    st.markdown('''<style>.css-v37k9u a {color: #ff4c4b;}</style>''',
        unsafe_allow_html=True)  # darkmode
    st.markdown('''<style>.css-nlntq9 a {color: #ff4c4b;}</style>''',
        unsafe_allow_html=True)  # lightmode
    st.markdown('Implemented by '
        '[N Bouyaa KASSINGA](https://www.linkedin.com/in/n-bouyaa-kassinga-818a02169/) - '
        'view project source code on '
        '[GitHub](https://github.com/FadelT/ml_avance_ok)')
    st.write('\n\n')

    options = st.sidebar.selectbox('What task do you wish?', ('regression', 'classification'),key=1)
    if options=='regression':
            choix_base_donnee = st.sidebar.selectbox('choose your regression dataset', ('custom_dataset', 'sklearn_diabetes'),key=2)
            if choix_base_donnee=='custom_dataset':
               nb_rows = st.sidebar.number_input("Write number of rows: ",500,1000000,500,100)
               nb_cols=st.sidebar.number_input("Write number of cols:",2,20,5,1)
               nb_rows,nb_cols=int(nb_rows),int(nb_cols)
               data=generer_base_donnee(nature='regression',nb_row=nb_rows,nb_cols=nb_cols,nb_class=None)
            elif choix_base_donnee=='sklearn_diabetes':
                data=use_sklearn_datasets(donnee='diabetes')
                print(type(data[0]))
                data=pd.concat([data[0],data[1]],axis=1)
               

    elif options=='classification':
           choix_base_donnee = st.sidebar.selectbox('choose your classification dataset', ('custom_dataset', 'sklearn_iris','sklearn_wine'),key=3)
           if choix_base_donnee=='custom_dataset':
               nb_rows = st.sidebar.number_input("Write number of rows: ",500,1000000,500,100)
               nb_cols=st.sidebar.number_input("Write number of cols:",2,20,5,1)
               nb_class=st.sidebar.number_input('Write number of classes:',2,2,2,1)
               nb_rows,nb_cols,nb_class=int(nb_rows),int(nb_cols),int(nb_class)
               data=generer_base_donnee(nature='classification',nb_row=nb_rows,nb_cols=nb_cols,nb_class=nb_class)
               data=np.concatenate((data[0],data[1].reshape((len(data[1]),1))),axis=1)

               print(data)
           elif choix_base_donnee=='sklearn_iris':
                data=use_sklearn_datasets(donnee='iris')
                #print(data['target'].shape)
                data=np.concatenate((data['data'],data['target'].reshape((len(data['target']),1))),axis=1)
                #data=pd.concat([data[0],data[1]],axis=1)
           elif choix_base_donnee=='sklearn_wine':
                data=use_sklearn_datasets(donnee='wine')
                #print(data)
                data=np.concatenate((data['data'],data['target'].reshape((len(data['target']),1))),axis=1)
                #data=pd.concat([data[0],data[1]],axis=1)
    if isinstance(data,np.ndarray):
        cols=[]
        for i in range(data.shape[1]):
            cols.append('col_{}'.format(i))
        data = pd.DataFrame(data, columns = cols)
    print(data)
    fig, ax = plt.subplots()
    descri=caracteristics_data(data)
    st.header('Description of your data:')
    st.table( descri)
    sns.heatmap(data.corr(), ax=ax)
    st.header('Pairplot of your data')
    fig1 = sns.pairplot(data)
    st.pyplot(fig1)
    st.header('Correlation between variables')
    st.write(fig)
    
    st.write('Your initial data shape is: {}'.format((data.shape[0],data.shape[1])))
    st.write('You selected {}'.format(options))
    trunc=st.sidebar.selectbox("Do you want to truncate your data before applying ML_models?",("Yes", "NO"))
    if trunc=="Yes":
        nb_rows = st.sidebar.number_input("Write number of rows: ",100,len(data),100,50)
        #nb_cols=st.sidebar.number_input("Write number of cols:",2,nb_cols,5,1)
        data=data.iloc[:int(nb_rows),:]

    normal_tech=st.sidebar.selectbox('Choose your features normalization technic:',('standardScaler','MinMax','Normal'))
    X=data.iloc[:,:-1]
    if choix_base_donnee!='sklearn_diabetes':
        X=normaliser(data.iloc[:,:-1],normal_tech)
    
        
    y=data.iloc[:,-1]
    parametres_model={}
    if options=='regression':
     model_name=st.sidebar.selectbox('choose your regression model',('LinearRegression','KNeighborRegressor','XgboostRegressor'))
    elif options=='classification':
     model_name=st.sidebar.selectbox('choose your classification model',('LogisticRegression','KNeighborclassifier','XgboostClassifier'))
    parametres_model['name']=model_name
    if model_name=='KNeighborRegressor' or model_name=='KNeighborclassifier':
        parametres_model['nb_neighbors'] = st.sidebar.number_input("Write number of neighbors: ",1,40,3,1)
        
    elif model_name=='XgboostRegressor' or model_name=='XgboostClassifier':
            parametres_model['n_estimators'] = st.sidebar.number_input("Set the  number of estimators: ",5,100,5,1)
            parametres_model['max_depth'] = st.sidebar.number_input("Set  max_depth: ",0,40,3,1)
            parametres_model['eta'] = st.sidebar.number_input("Set  eta: ",0.0,1.0,0.3,0.1)
            parametres_model['subsample']=st.sidebar.number_input("Set  subsample: ",0.0,1.0,0.5,0.1)
            parametres_model['colsample_bytree']=st.sidebar.number_input("Set  colsample_bytree: ",0.0,1.0,0.5,0.1)


    elif model_name=='LogisticRegression':
             penalty=st.sidebar.selectbox('choose your penalty method',('l1','l2'))
             parametres_model['penalty']=penalty
             solver_tuple=('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')

             if penalty=='l1':
                 solver_tuple=('liblinear','saga')
             parametres_model['solver']=st.sidebar.selectbox('choose your solver method',solver_tuple)
    score=entrain_algo(options,X,y,0.15,parametres_model)
    st.write("Model Performance on test data: {}".format(score))
             
    pass


if __name__ == '__main__':
    
    
    # call main function
    start_job()
