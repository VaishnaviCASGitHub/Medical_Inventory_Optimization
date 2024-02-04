# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:08:09 2024

@author: C Adinarayana
"""
############################################### PROBLEM DESCRIPTION ########################################################
'''
###### Business Problem: Bounce rate is increasing significantly leading to patient dissatisfaction.
###### Business Objective: Minimize bounce rate.
###### Business Constraint: Minimize inventory cost.
Success Criteria:
###### Business Success Criteria: Reduce bounce rate by atleast 30%
###### Economic Success Criteria: Increase revenue by atleast 20lacs INR by reducing bounce rate.
###### ML Success Criteria: Achieve an accuracy of atleast 90%
'''
###########################################################################################################################
 

import os # To manage software and hardware on the computer.
os.chdir(r"D:\VAISH 360 ASSIGNMENTS\MEDICAL INVENTORY PROJ INNO\Python EDA Code with dataset")
os.getcwd()

################################################ Importing Libraries ######################################################

import pandas as pd # Used of for Data Manipulation.
import numpy as np # used for numerical calculation.
import matplotlib.pyplot as plt # Used for Data Visualization
import seaborn as sns # used for Data Visualization
import datetime as dt # handle the timeseries data
import calendar # to handle date type data 
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,RobustScaler # to perform encoding
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import pickle

inventory = pd.read_excel(r"D:\VAISH 360 ASSIGNMENTS\MEDICAL INVENTORY PROJ INNO\Python EDA Code with dataset\Medical Inventory Optimization Dataset.xlsx")
inventory.info() 

################################################### DATA PREPROCESSING ####################################################

################## 1.Type Casting ##################

# * The Patient_ID column is categorical but given as Numerical.
inventory['Patient_ID'] = inventory['Patient_ID'].astype(str)

# * The Dateofbill is also time series but given as object we need to change.
inventory['Dateofbill'] = inventory['Dateofbill'].astype('datetime64')

################## 2. Handling Duplicates ##################

inventory.duplicated().sum()

inventory.drop_duplicates(inplace = True,keep = 'first')

################## 3. Adding of columns for EDA #################

inventory['bill_of_month'] = inventory['Dateofbill'].dt.strftime('%B')


inventory['bill_of_quater'] = inventory['Dateofbill'].dt.quarter

################### 4. EDA & Preprocessing Forecasting for Week ######################

inventory.sort_values(by ='Dateofbill',inplace=True)
inventory['weekofbill'] = inventory['Dateofbill'].dt.isocalendar().week
inventory['week_of_first_day'] = inventory.groupby('weekofbill')['Dateofbill'].transform('min').dt.isocalendar().week
inventory_week = inventory.groupby('weekofbill')['Quantity'].sum().reset_index()


plt.plot(inventory_week.index, inventory_week['Quantity'])
plt.show()

# Plot autocorrelation using statsmodels.graphics.tsaplots.plot_acf
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

plot_acf(inventory_week['Quantity'], lags=25) # Assuming monthly data, set lags to 25 for one year
plt.title('Autocorrelation Plot for Quantity in 2022')
plt.xlabel('Lags (Months)')
plt.ylabel('Autocorrelation')
plt.show()
plot_pacf(inventory_week['Quantity'], lags=25)


# Stationarity Check:

'''A stationary time series is one whose statistical properties, such as mean, variance, and autocorrelation, 
do not change over time. In contrast, a non-stationary time series might exhibit trends, seasonality, or other 
patterns that make its statistical properties vary over time.'''

# The null hypothesis of the ADF(Augumented Decay Fuller) test is that the time series is non-stationary. 

# Null Hypothesis (H0): The time series is non-stationary.
# Alternative Hypothesis (H1): The time series is stationary.


from statsmodels.tsa.stattools import adfuller
dftest = adfuller(inventory_week.Quantity , autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():
    print("\t",key, ": ", val)

'''To know more about the time series stationarity, we can perform the ADfuller test, a test based on hypothesis, 
where if the p-value is less than 0.05, then we can consider the time series is stationary, and if the P-value is
greater than 0.05, then the time series is non-stationary.'''

# ----------------------------------------------------- MOVING AVERAGE ----------------------------------------------------#

ma_pred = inventory_week['Quantity'].rolling(5).mean()
MA = metrics.mean_absolute_percentage_error(ma_pred[4:], inventory_week['Quantity'][4:])
MA 

# ----------------------------------------------------- SARIMAX MODEL ------------------------------------------------------#

# Based on the adfuller model we can say that our data is not a stationary.
# So we are using the SARIMAX model. 

import statsmodels.api as sm

model=sm.tsa.statespace.SARIMAX(inventory_week['Quantity'],order=(4, 1, 10),seasonal_order=(1,0,1,52))
results=model.fit()

inventory_week['forecast']=results.predict(start=0,end=52,dynamic=False)
inventory_week[['Quantity','forecast']][1:].plot(figsize=(12,8))
np.sqrt(metrics.mean_squared_error(inventory_week['forecast'][1:],inventory_week['Quantity'][1:]))

sari = metrics.mean_absolute_percentage_error(inventory_week['Quantity'][1:], inventory_week['forecast'][1:])

# ----------------------------------------------------- ARIMA MODEL -------------------------------------------------------#

from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(inventory_week['Quantity'], order=(4,1,10))
results = model.fit()

print(results.summary())

forecast = results.predict(start = 1,end = 51)

inventory_week[['Quantity']].plot(figsize=(12,8))
forecast.plot(figsize=(12,8))
rmse = np.sqrt(metrics.mean_squared_error(inventory_week['Quantity'][1:], forecast))

ari = metrics.mean_absolute_percentage_error(inventory_week['Quantity'][1:], forecast)

# ----------------------------------------------------- SARIMA MODEL ------------------------------------------------------#

"""
SARIMA introduces the concept of seasonality to the ARIMA model. 
Seasonality refers to recurring patterns or cycles in the data that repeat at
regular intervals. The seasonal component is denoted by the parameter S, 
which represents the number of observations in each season.

p: Order of the autoregressive (AR) component.
d: Degree of differencing.
q: Order of the moving average (MA) component.
P: Seasonal order of the autoregressive (AR) component.
D: Seasonal degree of differencing.
Q: Seasonal order of the moving average (MA) component.
s: Seasonal period.
"""
sarimax_data = inventory.drop(['Patient_ID','bill_of_month','bill_of_quater'],axis =1)

sarimax_data.drop(['Typeofsales','DrugName','RtnMRP','Final_Sales','Dateofbill'], axis =1, inplace=True)
# sarimax_data.drop(['SubCat'], axis =1, inplace=True) When we drop this feature i got MAPE = 5.89% 
# and it not drops this i got the 2 approximately
sarimax_data.drop(['Specialisation'], axis =1, inplace=True)
sarimax_data.drop(['Quantity'], axis =1, inplace=True)
sarimax_data.info()

numerical = sarimax_data.select_dtypes(['int64','float64']).columns
categorical = sarimax_data.select_dtypes(['object']).columns
num = Pipeline(steps= [('scaling',RobustScaler())])
cat = Pipeline([('encoding',OneHotEncoder())])
preprocess = ColumnTransformer([('scaling',num,numerical),
                                ('encoding',cat,categorical)],remainder=  'passthrough')
preprocess_fit =  preprocess.fit(sarimax_data)


sarimax_data_preprocess = pd.DataFrame(preprocess_fit.transform(sarimax_data).toarray() ,columns=preprocess_fit.get_feature_names_out())

sarimax_data = pd.concat([sarimax_data_preprocess,inventory[['Dateofbill','Quantity']]],axis =1)

sarimax_data['weekofbill'] = sarimax_data['Dateofbill'].dt.isocalendar().week
inventory['week_of_first_day'] = sarimax_data.groupby('weekofbill')['Dateofbill'].transform('min').dt.isocalendar().week
sarimax_data = sarimax_data.groupby('weekofbill').sum().reset_index()



train = sarimax_data
test = sarimax_data
from statsmodels.tsa.statespace.sarimax import SARIMAX
sarimax_model = SARIMAX(train['Quantity'],exog=train.drop(['Quantity','weekofbill'],axis=1), order=(4, 1, 10),seasonal_order=(1,0,1,52))
sarimax_result = sarimax_model.fit()
pred_sarimax = sarimax_result.predict(start = test.index[0], end = test.index[-1])
sarimax = metrics.mean_absolute_percentage_error(pred_sarimax, test.Quantity)
sarimax*100

# ----------------------------------------------------- SES MODEL ---------------------------------------------------------#

from statsmodels.tsa.holtwinters import SimpleExpSmoothing

dict1 = {}
for alpha in range(1,10):
    model = SimpleExpSmoothing(inventory_week['Quantity'])
    model_fit = model.fit(smoothing_level=alpha/10, optimized=False)
    predictions = model_fit.predict(start=0, end=len(inventory_week['Quantity'])-1)
    mape = metrics.mean_absolute_percentage_error(inventory_week['Quantity'], predictions)
    dict1[alpha/10] = mape
ses = min(dict1.values())

# ----------------------------------------------------- HOLT-WINTER'S MODEL -----------------------------------------------#

from statsmodels.tsa.holtwinters import ExponentialSmoothing

list_holt = []
# Define the hyperparameter combinations
def holt_winters(trend, seasonal, periods, smooth, use_boxcox):
    model = ExponentialSmoothing(inventory_week['Quantity'], trend=trend, seasonal=seasonal,
                                 seasonal_periods=periods, use_boxcox=use_boxcox)
    model_fit = model.fit(smoothing_level=smooth, smoothing_trend=smooth, smoothing_seasonal=smooth)
    predictions = model_fit.predict(start=0, end=len(inventory_week['Quantity']) - 1)
    mape = metrics.mean_absolute_percentage_error(inventory_week['Quantity'], predictions)
    return mape

# Specify the hyperparameter grid
trend_seasonal = ['add', 'additive', 'mul', 'multiplicative']
seasonal_periods = [5, 7, 9, 11, 13, 15, 17, 19]
smoothing = [0.2, 0.5, 0.8]
use_boxcox_values = [True, False]  # Set to True or False based on your preference

# Iterate over hyperparameter combinations
for trend in trend_seasonal:
    for seasonal in trend_seasonal:
        for periods in seasonal_periods:
            for smooth in smoothing:
                for use_boxcox in use_boxcox_values:
                    mape = holt_winters(trend, seasonal, periods, smooth, use_boxcox)
                    print(f"Parameters: Trend={trend}, Seasonal={seasonal}, Periods={periods}, Smoothing={smooth}, Use_BoxCox={use_boxcox}, MAPE={mape}")
                    list_holt.append(mape)
hw = min(list_holt)
# This are the best parametres for this model 
# Parameters: Trend=multiplicative, Seasonal=multiplicative, Periods=15, Smoothing=0.2, 
# Use_BoxCox=False, MAPE=0.13376882169095172

# ----------------------------------------------------- VAR MODEL --------------------------------------------------#

from statsmodels.tsa.api import VAR

inventory_week['log_Quantity'] = np.log(inventory_week['Quantity'])
inventory_week['week_square'] = np.square(inventory_week['weekofbill'])
inventory_week.week_square = inventory_week.week_square.astype('int64')
inventory_week.info()
VAR_data = inventory_week[['Quantity','week_square', 'log_Quantity']]

model = VAR(VAR_data)
model_fitted = model.fit()
lag_order = model_fitted.k_ar

forecast = model_fitted.forecast(VAR_data.values[-lag_order:], steps=10)
VAR_mape = metrics.mean_absolute_percentage_error(VAR_data[-10:], forecast)
VAR_mape

# ----------------------------------------------------- FTS MODEL -----------------------------------------------#

#pip install scikit-fuzzy
import skfuzzy as fuzz
from skfuzzy import control as ctrl

train = inventory_week
test = inventory_week

# Extract the target column
train_series = train['Quantity'].values
test_series = test['Quantity'].values

# Define fuzzy sets and antecedents
antecedent = ctrl.Antecedent(np.arange(min(train_series), max(train_series), 1), 'input')
consequent = ctrl.Consequent(np.arange(min(train_series), max(train_series), 1), 'output')

# Create fuzzy membership functions
antecedent['low'] = fuzz.trimf(antecedent.universe, [min(train_series), min(train_series), np.median(train_series)])
antecedent['medium'] = fuzz.trimf(antecedent.universe, [min(train_series), np.median(train_series), max(train_series)])
antecedent['high'] = fuzz.trimf(antecedent.universe, [np.median(train_series), max(train_series), max(train_series)])

consequent['low'] = fuzz.trimf(consequent.universe, [min(train_series), min(train_series), np.median(train_series)])
consequent['medium'] = fuzz.trimf(consequent.universe, [min(train_series), np.median(train_series), max(train_series)])
consequent['high'] = fuzz.trimf(consequent.universe, [np.median(train_series), max(train_series), max(train_series)])

# Create fuzzy rules
rule1 = ctrl.Rule(antecedent['low'], consequent['low'])
rule2 = ctrl.Rule(antecedent['medium'], consequent['medium'])
rule3 = ctrl.Rule(antecedent['high'], consequent['high'])

# Create Fuzzy Control System
fts_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
fts_simulation = ctrl.ControlSystemSimulation(fts_ctrl)

# Fit the model with training data
fts_simulation.input['input'] = train_series
fts_simulation.compute()

# Make predictions on test data
fts_predictions = fts_simulation.output['output']

# Calculate MAPE
FTS_mape = metrics.mean_absolute_percentage_error(test_series, fts_predictions)
FTS_mape # 10.280206975315044

# ----------------------------------------------------- GRU MODEL ---------------------------------------------#

from sklearn.preprocessing import MinMaxScaler

train_series = train['Quantity'].values
test_series = test['Quantity'].values

# Normalize the data
scaler = MinMaxScaler()
train_series_scaled = scaler.fit_transform(train_series.reshape(-1, 1))
test_series_scaled = scaler.transform(test_series.reshape(-1, 1))
# Define a function to create sequences for training
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)

sequence_length = 10  
X_train = create_sequences(train_series_scaled, sequence_length)
y_train = train_series_scaled[sequence_length:]

#pip install tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

# Create a GRU model
model = Sequential()
model.add(GRU(units=50, input_shape=(sequence_length, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=50, batch_size=32)

X_test = create_sequences(test_series_scaled, sequence_length)

predictions_scaled = model.predict(X_test)

predictions = scaler.inverse_transform(predictions_scaled)

GRU_mape = metrics.mean_absolute_percentage_error(test_series[sequence_length:], predictions.flatten())
GRU_mape  # 12.619479277012296

# ----------------------------------------------------- LSTM MODEL ---------------------------------------------#

from sklearn.preprocessing import MinMaxScaler

train_series = train['Quantity'].values
test_series = test['Quantity'].values

# Normalize the data
scaler = MinMaxScaler()
train_series_scaled = scaler.fit_transform(train_series.reshape(-1, 1))
test_series_scaled = scaler.transform(test_series.reshape(-1, 1))

# Define a function to create sequences for training
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)

sequence_length = 10  
X_train = create_sequences(train_series_scaled, sequence_length)
y_train = train_series_scaled[sequence_length:]

#pip install tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM

# Create a LSTM model
model = Sequential()
model.add(LSTM(units=50, input_shape=(sequence_length, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=10, batch_size=10)

X_test = create_sequences(test_series_scaled, sequence_length)

predictions_scaled = model.predict(X_test)

predictions = scaler.inverse_transform(predictions_scaled)

LSTM_mape = metrics.mean_absolute_percentage_error(test_series[sequence_length:], predictions.flatten())
LSTM_mape   # 12.743400225674499

# ----------------------------------------------------- RNN MODEL ---------------------------------------------#

from sklearn.preprocessing import MinMaxScaler

train_series = train['Quantity'].values
test_series = test['Quantity'].values

# Normalize the data
scaler = MinMaxScaler()
train_series_scaled = scaler.fit_transform(train_series.reshape(-1, 1))
test_series_scaled = scaler.transform(test_series.reshape(-1, 1))

# Define a function to create sequences for training
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)

sequence_length = 10  
X_train = create_sequences(train_series_scaled, sequence_length)
y_train = train_series_scaled[sequence_length:]

#pip install tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN

# Create a LSTM model
model = Sequential()
model.add(SimpleRNN(units=50, input_shape=(sequence_length, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=10, batch_size=10)

X_test = create_sequences(test_series_scaled, sequence_length)

predictions_scaled = model.predict(X_test)

predictions = scaler.inverse_transform(predictions_scaled)

RNN_mape = metrics.mean_absolute_percentage_error(test_series[sequence_length:], predictions.flatten())
RNN_mape    # 11.183624222872476

#######################################------------ COMPARING MAPE --------------------#################################################

mi = pd.Series({'Moving Average':MA,'Simple Exponential':ses,'Holts Winter':hw,
                'ARIMA':ari,'SARIMA':sari,'SARIMAX':sarimax,'FTS':FTS_mape,'GRU':GRU_mape,'LSTM':LSTM_mape,'RNN':RNN_mape,'VAR':VAR_mape})
mape = pd.DataFrame(mi, columns=['mape'])
mape

#######################################------------ BEST MODEL -----------------------#################################################

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@--------------SARIMAX MODEL --------------------@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #

import pandas as pd
import datetime as dt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler,OneHotEncoder
import joblib
from sklearn import metrics
from sklearn.compose import ColumnTransformer

inventory = pd.read_excel(r"D:\VAISH 360 ASSIGNMENTS\MEDICAL INVENTORY PROJ INNO\Python EDA Code with dataset\Medical Inventory Optimization Dataset.xlsx")

#################################### Type Casting ##########################################

inventory.info()

# 1) The ID varialbe is categorical but it given as Numerical.
# 2) The Dateofbill is also also time series but given as object we need to change.

inventory['Patient_ID'] = inventory['Patient_ID'].astype(str)

inventory['Dateofbill'] = inventory['Dateofbill'].astype('datetime64[ns]')

################################### Handling Duplicates #################################

inventory.duplicated().sum()

inventory.drop_duplicates(inplace = True,keep = 'first')

sarimax_data = inventory.drop(['Patient_ID'],axis =1)

inventory['weekofbill'] = inventory['Dateofbill'].dt.isocalendar().week

sarimax_data.drop(['Typeofsales','DrugName','RtnMRP','Final_Sales','Dateofbill'], axis =1, inplace=True)
sarimax_data.drop(['SubCat'], axis =1, inplace=True) #When we drop this feature i got MAPE = 5.89% 
# and it not drops this i got the 2 approximately
sarimax_data.drop(['Specialisation'], axis =1, inplace=True)
sarimax_data.drop(['Quantity'], axis =1, inplace=True)
sarimax_data.info()

numerical = sarimax_data.select_dtypes(['int64','float64']).columns
categorical = sarimax_data.select_dtypes(['object']).columns
num = Pipeline(steps= [('scaling',RobustScaler())])
cat = Pipeline([('encoding',OneHotEncoder())])
preprocess = ColumnTransformer([('scaling',num,numerical),
                                ('encoding',cat,categorical)],remainder=  'passthrough')
preprocess_fit =  preprocess.fit(sarimax_data)


sarimax_data_preprocess = pd.DataFrame(preprocess_fit.transform(sarimax_data).toarray() ,columns=preprocess_fit.get_feature_names_out())

sarimax_data = pd.concat([sarimax_data_preprocess,inventory[['weekofbill','Quantity']]],axis =1)

sarimax_data = sarimax_data.groupby('weekofbill').sum().reset_index()



train = sarimax_data
test = sarimax_data
from statsmodels.tsa.statespace.sarimax import SARIMAX
sarimax_model = SARIMAX(train['Quantity'],exog=train.drop(['Quantity','weekofbill'],axis=1), order=(4, 1, 10),seasonal_order=(1,0,1,52))
sarimax_result = sarimax_model.fit()
pred_sarimax = sarimax_result.predict(start = test.index[0], end = test.index[-1])
sarimax = metrics.mean_absolute_percentage_error(pred_sarimax, test.Quantity)
root_mean = np.sqrt(metrics.mean_squared_error(pred_sarimax,train['Quantity']))
sarimax*100



best_model = sarimax_result
joblib.dump(best_model, 'best_model_sarimax.pkl')
joblib.dump(preprocess_fit,'preprocess_fit.pkl')

##################################################### Deployment Code ##########################################################

import joblib
import os 
import pandas as pd # Used of for Data Manipulation.
import numpy as np # used for numerical calculation.
import datetime as dt # handle the timeseries data.

os.chdir(r'D:\VAISH 360 ASSIGNMENTS\MEDICAL INVENTORY PROJ INNO\Python EDA Code with dataset')

sarimax_model = joblib.load('best_model_sarimax.pkl') 
preprocess_fit = joblib.load('preprocess_fit.pkl')

inventory['Patient_ID'] = inventory['Patient_ID'].astype(str)

inventory['Dateofbill'] = inventory['Dateofbill'].astype('datetime64[ns]')

################################### Handling Duplicates #################################

inventory.duplicated().sum()

inventory.drop_duplicates(inplace = True,keep = 'first')

sarimax_data = inventory.drop(['Patient_ID'],axis =1)

inventory['weekofbill'] = inventory['Dateofbill'].dt.isocalendar().week

sarimax_data.drop(['Typeofsales','DrugName','RtnMRP','Final_Sales','Dateofbill'], axis =1, inplace=True)
sarimax_data.drop(['SubCat'], axis =1, inplace=True)#  When we drop this feature i got MAPE = 5.89% 
# and it not drops this i got the 2 approximately
sarimax_data.drop(['Specialisation'], axis =1, inplace=True)
sarimax_data.drop(['Quantity'], axis =1, inplace=True)

sarimax_data_preprocess = pd.DataFrame(preprocess_fit.transform(sarimax_data).toarray() ,columns=preprocess_fit.get_feature_names_out())


sarimax_data = pd.concat([sarimax_data_preprocess,inventory[['weekofbill','Quantity']]],axis =1)

sarimax_data = sarimax_data.groupby('weekofbill').sum().reset_index()

pred_sarimax = sarimax_model.predict(start = test.index[0], end = test.index[-1])

results = pd.concat([sarimax_data[['weekofbill','Quantity']],pred_sarimax],axis = 1)
results.columns = ['weekofbill','Actual_Quantity','Predicted_Quantity']
