import pandas as pd
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pickle
tf.random.set_seed(123)
np.random.seed(123)


df = pd.read_csv("FINAL_USO.csv") # we found one file  FInal.csv after unzipping fetching it into data frame

df = df.drop_duplicates(keep= 'first')

df['Date'] = pd.to_datetime(df['Date'])

df.sort_values(by='Date',ascending=True, inplace=True)
st.subheader('Description of the Dataset')
st.write(df.describe())

import matplotlib as mpl
# gloabl params for all matplotlib plots 
mpl.rcParams['figure.figsize'] = (15, 12)
mpl.rcParams['axes.grid'] = False

# with the following function we can select highly correlated features
# it will remove the first feature that is correlated with anything other feature

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation(df, 0.80)  # calling the orrelation function
len(set(corr_features))

l =list(corr_features)
#correlated features
st.subheader('Correlated Features')
st.write(print(l))

df1 =df # saving a copy of original dataframe,as we will need the target variable which is Adj Close, but we are going to drop it from df 
#df = df.drop("Adj Close",axis=1)
df1.shape

df=df.drop(corr_features,axis=1) # dropping all the correlated features which are more than 85% correlated, so that out model does not fall into dummy variable trap

df.shape

l=list(df.columns)


df.columns

df=df.drop(['Date'],axis=1) # dropping Date as it can not be used in numerical calculation

df = df[[ 'Open', 'Volume', 'SP_open', 'SP_volume', 'DJ_volume',
       'EG_volume', 'EU_Price', 'EU_Trend', 'OF_Volume', 'OF_Trend',
       'OS_Trend', 'SF_Volume', 'SF_Trend', 'USB_Price', 'USB_Trend',
       'PLT_Trend', 'PLD_Price', 'PLD_Trend', 'RHO_PRICE', 'USDI_Volume',
       'USDI_Trend', 'GDX_Volume', 'USO_Volume']]

def custom_ts_multi_data_prep(dataset, target, start, end, window, horizon):
    X = []
    y = []
    start = start + window
    if end is None:
        end = len(dataset) - horizon

    for i in range(start, end):
        indices = range(i-window, i)
        X.append(dataset[indices])

        indicey = range(i+1, i+1+horizon)
        y.append(target[indicey])
    return np.array(X), np.array(y)

validate = df1[[ 'Open', 'Volume', 'SP_open', 'SP_volume', 'DJ_volume',
       'EG_volume', 'EU_Price', 'EU_Trend', 'OF_Volume', 'OF_Trend',
       'OS_Trend', 'SF_Volume', 'SF_Trend', 'USB_Price', 'USB_Trend',
       'PLT_Trend', 'PLD_Price', 'PLD_Trend', 'RHO_PRICE', 'USDI_Volume',
       'USDI_Trend', 'GDX_Volume', 'USO_Volume','Adj Close']].tail(14) #use number same as forecastr horizon 
df.drop(df.tail(516).index,inplace=True)# this drop was not needed to do on 516 rows, rather only drop number of rows = forecast horizaon, so our training and testing data points will increase instead of 1202 we can use len(df)-14

x_scaler = preprocessing.MinMaxScaler()
y_scaler = preprocessing.MinMaxScaler()
dataX = x_scaler.fit_transform(df[['Open', 'Volume', 'SP_open', 'SP_volume', 'DJ_volume',
       'EG_volume', 'EU_Price', 'EU_Trend', 'OF_Volume', 'OF_Trend',
       'OS_Trend', 'SF_Volume', 'SF_Trend', 'USB_Price', 'USB_Trend',
       'PLT_Trend', 'PLD_Price', 'PLD_Trend', 'RHO_PRICE', 'USDI_Volume',
       'USDI_Trend', 'GDX_Volume', 'USO_Volume']])#
dataY = y_scaler.fit_transform(df1[['Adj Close']])#df1

hist_window = 90
horizon = 14 # this value is significant, validation, splitting all depends on it
TRAIN_SPLIT = 960
x_train_multi, y_train_multi = custom_ts_multi_data_prep(
    dataX, dataY, 0, TRAIN_SPLIT, hist_window, horizon)
x_val_multi, y_val_multi = custom_ts_multi_data_prep(
    dataX, dataY, TRAIN_SPLIT, None, hist_window, horizon)

print ('Single window of past history')
print(len(x_train_multi[0]))
print(x_train_multi[0])

print('\n Target horizon')
print(len(y_train_multi[0]))
print(y_train_multi[0])

BATCH_SIZE = 256
BUFFER_SIZE = 300

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

model_path = r'\Tests_bidirec\Bidirectional_LSTM_Multivariate_Gold.h5'

model_bi = pickle.load(open('/content/trained_model.sav','rb'))


data_val = x_scaler.fit_transform(df1[['Open', 'Volume', 'SP_open', 'SP_volume', 'DJ_volume',
       'EG_volume', 'EU_Price', 'EU_Trend', 'OF_Volume', 'OF_Trend',
       'OS_Trend', 'SF_Volume', 'SF_Trend', 'USB_Price', 'USB_Trend',
       'PLT_Trend', 'PLD_Price', 'PLD_Trend', 'RHO_PRICE', 'USDI_Volume',
       'USDI_Trend', 'GDX_Volume', 'USO_Volume']].tail(90))

val_rescaled = data_val.reshape(1, data_val.shape[0], data_val.shape[1])

Predicted_results = model_bi.predict(val_rescaled)
Predicted_results_Inv_trans = y_scaler.inverse_transform(Predicted_results)

st.subheader('Model predictions')
st.write(Predicted_results_Inv_trans)

from sklearn import metrics
def timeseries_evaluation_metrics_func(y_true, y_pred):
    
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print('Evaluation metric results:-')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}',end='\n\n')


timeseries_evaluation_metrics_func(validate['Adj Close'],Predicted_results_Inv_trans[0])
st.subheader('Actuals Vs. Predictions')
fig = plt.figure(figsize=(6,4))
plt.plot( list(validate['Adj Close']))
plt.plot( list(Predicted_results_Inv_trans[0]))
plt.title("Actual vs Predicted")
plt.ylabel('Adj Close')
plt.legend(('Actual','predicted'))
plt.show()
st.pyplot(fig)

train_results = pd.DataFrame(data={'Predictions':Predicted_results_Inv_trans[0], 'Actuals':validate['Adj Close']})
train_results

