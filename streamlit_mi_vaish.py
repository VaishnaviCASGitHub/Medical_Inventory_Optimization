# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:50:49 2024

@author: C Adinarayana
"""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import numpy as np  # used for numerical calculation.
import datetime as dt  # handle the timeseries data
import os
import statsmodels.api as sm
import sklearn
import joblib
import time
os.chdir(r'D:\VAISH 360 ASSIGNMENTS\MEDICAL INVENTORY PROJ INNO\Python EDA Code with dataset')
sarimax_model = joblib.load(r"D:\VAISH 360 ASSIGNMENTS\MEDICAL INVENTORY PROJ INNO\Python EDA Code with dataset\best_model_sarimax.pkl")
preprocess_fit = joblib.load(r"D:\VAISH 360 ASSIGNMENTS\MEDICAL INVENTORY PROJ INNO\Python EDA Code with dataset\preprocess_fit.pkl")
os.getcwd()
def main():

    st.title(":grey[INNODATATICS Project]", anchor=None)
    st.sidebar.title(":rainbow[Forecasting Model]")

    html_temp = """
    <div style="background-color:#F72585;padding:10px">
    <h2 style="color:white;text-align:center;">Optimization of Medical Inventory using Forecasting </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.text("")

    uploadedFile = st.sidebar.file_uploader("Choose a file", type=[
                                            'csv', 'xlsx'], accept_multiple_files=False, key="fileUploader")
    
    if uploadedFile is not None:
 
        if uploadedFile is not None:
            try:
    
                data = pd.read_csv(uploadedFile)
            except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:
                    data = pd.DataFrame(uploadedFile)

    else:
        st.sidebar.warning("you need to upload a csv or excel file.")

    html_temp = """
    <div style="background-color:#4CC9F0;padding:5px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html=True)

    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here", type="password")
    db = st.sidebar.text_input("database", "Type Here")

    if st.button(":blue[Predict]"):
        
        engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}") 
        
        inventory = data
        with st.spinner('Wait for the forecast results...'):
                 time.sleep(5)
                 st.success('Your results!')
        ###############################################
        st.subheader(":grey[Forecasted data]", anchor=None)

        inventory['Patient_ID'] = inventory['Patient_ID'].astype(str)

        inventory['Dateofbill'] = inventory['Dateofbill'].astype(
            'datetime64[ns]')

        ################################### Handling Duplicates #################################
        inventory.duplicated().sum()

        inventory.drop_duplicates(inplace=True, keep='first')
       
        sarimax_data = inventory.drop(['Patient_ID'], axis=1)

        inventory['weekofbill'] = inventory['Dateofbill'].dt.isocalendar().week

        sarimax_data.drop(['Typeofsales', 'DrugName', 'RtnMRP',
                          'Final_Sales', 'Dateofbill'], axis=1, inplace=True)
        # When we drop this feature i got MAPE = 5.89%
        sarimax_data.drop(['SubCat'], axis=1, inplace=True)
        # and it not drops this i got the 2 approximately
        sarimax_data.drop(['Specialisation'], axis=1, inplace=True)
        sarimax_data.drop(['Quantity'], axis=1, inplace=True)

        sarimax_data_preprocess = pd.DataFrame(preprocess_fit.transform(
            sarimax_data).toarray(), columns=preprocess_fit.get_feature_names_out())

        sarimax_data = pd.concat(
            [sarimax_data_preprocess, inventory[['weekofbill', 'Quantity']]], axis=1)

        sarimax_data = sarimax_data.groupby('weekofbill').sum().reset_index()

        pred_sarimax = sarimax_model.predict(
            start=sarimax_data.index[0], end=sarimax_data.index[-1])
        # , exog=train.drop(['Quantity', 'weekofbill'], axis=1)

        # So here we are applying for the old data i am commenting.

        results = pd.concat([sarimax_data[['weekofbill','Quantity']],pred_sarimax],axis =1)
        
        
        results.columns = ['weekofbill','Actual_Quantity','Predicted_Quantity']
        results['Actual_Quantity']=results['Actual_Quantity'].astype(int)
        
        results['Predicted_Quantity']=results['Predicted_Quantity'].astype(int)
        
        results.to_sql('forecast_results_dd', con=engine,
                      if_exists='replace', index=False, chunksize=1000)

        import seaborn as sns
        cm = sns.light_palette("#F72585", as_cmap=True)
                # Apply the background_gradient to the Styler object
        styled_df = results.style.background_gradient(cmap=cm,axis=None)

        # Set the precision for floating-point numbers
        styled_df = styled_df.format(precision=2)

        # Display the styled DataFrame in Streamlit
        st.table(styled_df)
        st.text("")
        st.subheader(
            ":grey[Forecasts (vs) Actual Outcomes Plot]", anchor=None)
        # plot forecasts against actual outcomes
        fig, ax = plt.subplots()
        ax.plot(sarimax_data.Quantity, '-c', label='Actual Value')
        ax.plot(pred_sarimax, '--m', label='Predicted value')
        ax.legend()
        st.pyplot(fig)


if __name__ == '__main__':
    main()
