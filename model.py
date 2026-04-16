import numpy as np
import pandas as pd 
import os
import google.generativeai as genai
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import (f1_score, mean_squared_error,r2_score,
                            accuracy_score,
                            classification_report,confusion_matrix,
                            recall_score,precision_score)
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier,
                            GradientBoostingRegressor,RandomForestRegressor)

from analysis import suggest_improvements, generate_summary
from dotenv import load_dotenv
load_dotenv()

key=os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=key)
model=genai.GenerativeModel('gemini-2.5-flash-lite')


st.set_page_config(page_title='ML Models Demo', page_icon='🤖', layout='wide')
st.title(':green[ML model Automation] 📊🤖')
st.header('Streamlit App to get CSV and target as input and performs ML algorithms')


uploaded_file=st.file_uploader('Upload your document here 📝',type=['csv'])

if uploaded_file:
    st.markdown('##### Preview of the uploaded CSV file:')
    df=pd.read_csv(uploaded_file)
    st.write('Dataframe Preview:')
    st.dataframe(df.head())
    st.write('Dataframe Summary:')
    st.write(df.describe())  
    
    target = st.selectbox(':blue[Select the target variable]', df.columns)
    
    st.write(':red[You have selected the target variable:]', target)
    
    if target:
        
        X=df.drop(columns=[target]).copy()
        y=df[target].copy()
        
        # Preprocessing
        
        num_cols=X.select_dtypes(include=np.number).columns.tolist()
        cat_cols=X.select_dtypes(include='object').columns.tolist()
        
        # missing value imputation
        
        X[num_cols]=X[num_cols].fillna(X[num_cols].median())
        X[cat_cols]=X[cat_cols].fillna('Missing data')
        
        # encoding categorical variables
        X = pd.get_dummies(data=X, drop_first=True, columns=cat_cols,dtype=int)
        
        #for categorical target variable
        if y.dtype == 'object':
            label = LabelEncoder()
            y=label.fit_transform([y])
            
        # detect the problem type
        if df[target].dtype == 'object' or len(np.unique(y)) <= 20:
            problem_type = 'Classification'
            
        else:
            problem_type = 'Regression'
            
            
        st.write(':green[Detected problem type:]', problem_type)
        
        # split the data
        x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        
        # scale the data
        # fit_transform on train data
        # transform on test data
        
        for i in x_train.columns:
            s = StandardScaler()
            x_train[i] = s.fit_transform(x_train[[i]])
            x_test[i] = s.transform(x_test[[i]])
            
            
        #Models
        #==============================
        results=[]
        if problem_type == 'Regression':
            models = {'Linear Regression': LinearRegression(), 
                    'Random Forest': RandomForestRegressor(random_state=42),
                    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)}
            
            for name, model in models.items():
                model.fit(x_train,y_train)
                y_pred = model.predict(x_test)
                
                results.append({'Model Name':name,
                                'MSE': round(mean_squared_error(y_test,y_pred),3), 
                                'R2 score':round(r2_score(y_test,y_pred),3),
                                'RMSE': round(np.sqrt(mean_squared_error(y_test,y_pred)),3)})
                
        else:
            models = {'Logistic Regression': LogisticRegression(random_state=42), 
                    'Random Forest': RandomForestClassifier(random_state=42),
                    'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42)}
            
            for name, model in models.items():
                model.fit(x_train,y_train)
                y_pred = model.predict(x_test)
                
                results.append({'Model Name':name,
                                'Accuracy': round(accuracy_score(y_test,y_pred),3), 
                                'Recall':round(recall_score(y_test,y_pred,average='weighted'),3),
                                'Precision': round(precision_score(y_test,y_pred,average='weighted'),3),
                                'F1 Score': round(f1_score(y_test,y_pred,average='weighted'),3)})
        
        
        results_df = pd.DataFrame(results)
        st.write(':blue[Model Performance Comparison]')
        st.dataframe(results_df)
        
        if problem_type == 'Regression':
          st.bar_chart(results_df.set_index('Model Name')['R2 Score'])
          st.bar_chart(results_df.set_index('Model Name')['RMSE'])
        else:
          st.bar_chart(results_df.set_index('Model Name')['Accuracy'])
          st.bar_chart(results_df.set_index('Model Name')['F1 Score'])
          
        # AI Insights
        
        if st.button('Generate Summary'):
          summary = generate_summary(results_df)
          st.write(summary)
        
        if st.button('Suggest Improvements'):
          suggest = suggest_improvements(results_df)
          st.write(suggest)