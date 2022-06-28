#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[3]:


# import data
dt = pd.read_csv("C:\\Users\\admin\\Downloads\\EXECL R MATERIALS\\PROJECTS DETAILS\\Website Vistiors Daywise - Sheet1.csv")
dt


# # Step 1: Exploratory Data Analysis(EDA)

# In[4]:


# import data
data= pd.read_csv("C:\\Users\\admin\\Downloads\\EXECL R MATERIALS\\PROJECTS DETAILS\\Website Vistiors Daywise - Sheet1.csv", parse_dates=True, index_col="Date")
data


# In[5]:


data.head(5)


# In[6]:


data.tail()


# In[7]:


data.isna().sum()


# In[8]:


data.shape


# In[9]:


data.index


# In[10]:


data.describe()


# In[11]:


#printing date of max and min customer visited  
print(data.loc[data["Daily Visitors"]==data["Daily Visitors"].max()])
print(data.loc[data["Daily Visitors"]==data["Daily Visitors"].min()])


# # Step 2: Visualixation Of Data

# In[12]:


#plotting 
data.plot()


# In[13]:


data['Daily Visitors'].resample("M").mean().plot()   #resampling data Month wise


# In[14]:


data['Daily Visitors'].resample("Q").mean().plot()  #resampling data Quarter wise


# In[15]:


data['Daily Visitors'].resample("y").mean().plot(kind='bar')


# In[16]:


data.hist()


# In[17]:


data.plot(kind="kde")


# In[18]:


import seaborn as sns
sns.boxplot(x= data['Daily Visitors'])


# In[19]:


#Determine rolling statistics---->moving average smoothing is used to remove the noise

rolmean=data['Daily Visitors'].rolling(window=12).mean() #window size 12 denotes 12 raw observations useed to calculate the moving average

rolstd=data['Daily Visitors'].rolling(window=12).std()

print(rolmean,rolstd)


# In[20]:


#plot rolling statistics
orig=plt.plot(data['Daily Visitors'],color="blue",label="Original")
mean=plt.plot(rolmean,color="red",label="Rolling mean")
std=plt.plot(rolstd,color="black",label="Rolling std")
plt.legend(loc='best')
plt.title("Rolling Mean & Standard Deviation")
plt.show(block=False)


# In[21]:


#mean and standard deviation are not constant it means it has some seasonality



```
`# This is formatted as code`
```

#Determining rolling statistics
#here i am not sure abt this rolling statistics as i have resampled it month wise { because of this mean and std looks constant}
rolmean=data['Daily Visitors'].resample("M").mean().rolling(window=12).mean()

rolstd=data['Daily Visitors'].resample("M").mean().rolling(window=12).std()

print(rolmean,rolstd)# plot rolling statistics
orig=plt.plot(data['Daily Visitors'].resample("MS").mean(),color="blue",label="Original")
mean=plt.plot(rolmean,color="red",label="Rolling mean")
std=plt.plot(rolstd,color="black",label="Rolling std")
plt.legend(loc='best')
plt.title("Rolling Mean & Standard Deviation")
plt.show(block=False)
# In[22]:


#H0:data is not stationary
#H1:data is stationary
#To check wheather a time series is stationary or not we use dickey-fuller test
#Perform Augmented Dickey–Fuller test:
from statsmodels.tsa.stattools import adfuller
print('Results of Dickey Fuller Test:')
dftest = adfuller(data['Daily Visitors'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
    
print(dfoutput)


# Here p value is greater than 0.05 ,we fail to reject the null hypothesis that means time series not stationary.

# In[23]:


#using log transorfamation
data_logScale = np.log(data)
plt.plot(data_logScale)


# In[24]:


#The below transformation is required to make series stationary
movingAverage = data_logScale.rolling(window=12).mean()
movingSTD = data_logScale.rolling(window=12).std()
plt.plot(data_logScale)
plt.plot(movingAverage, color='red')


# In[25]:


datasetLogScaleMinusMovingAverage = data_logScale - movingAverage
datasetLogScaleMinusMovingAverage.head(12)

#Remove NAN values
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
datasetLogScaleMinusMovingAverage.head(10)


# In[26]:


def test_stationarity(timeseries):
    
    #Determine rolling statistics
    movingAverage = timeseries.rolling(window=12).mean()
    movingSTD = timeseries.rolling(window=12).std()
    
    #Plot rolling statistics
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(movingAverage, color='red', label='Rolling Mean')
    std = plt.plot(movingSTD, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey–Fuller test:
    print('Results of Dickey Fuller Test:')
    dftest = adfuller(timeseries['Daily Visitors'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    


# In[27]:


plt.figure(figsize=(15,6))
test_stationarity(datasetLogScaleMinusMovingAverage)

Here p value is < 0.05 ,now we can reject null hypothesis.
# In[28]:


exponentialDecayWeightedAverage = data_logScale.ewm(halflife=12, min_periods=0, adjust=True).mean()
plt.plot(data_logScale)
plt.plot(exponentialDecayWeightedAverage, color='red')


# In[29]:


datasetLogScaleMinusExponentialMovingAverage = data_logScale - exponentialDecayWeightedAverage
test_stationarity(datasetLogScaleMinusExponentialMovingAverage)


# In[30]:


datasetLogDiffShifting = data_logScale - data_logScale.shift()
plt.plot(datasetLogDiffShifting)


# In[31]:


datasetLogDiffShifting.dropna(inplace=True)
test_stationarity(datasetLogDiffShifting)


# In[32]:


from statsmodels.tsa.stattools import acf, pacf
#ACF & PACF plots

lag_acf = acf(datasetLogDiffShifting, nlags=20)
lag_pacf = pacf(datasetLogDiffShifting, nlags=20, method='ols')

#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')            

#Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
            
plt.tight_layout()            


# In[33]:


import statsmodels.api as sm


# In[34]:


import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(data["Daily Visitors"],lags=12)
tsa_plots.plot_pacf(data["Daily Visitors"],lags=12)
plt.show()


# In[35]:


#Plot PSD
plt.psd(data["Daily Visitors"], detrend='linear');
plt.title("PSD Plot");


# # Holt Winter’s

# In[36]:


#counting the number of missing values
df = (data['Daily Visitors']).isnull()
df


# In[37]:


#Outlier detection and treatment
plt.figure(figsize=(15,8))
sns.boxplot(x= data['Daily Visitors'])


# In[38]:


#calculating the z score
data['z_score'] = data['Daily Visitors'] - data['Daily Visitors'].mean()/data['Daily Visitors'].std(ddof=0)


# In[39]:


from scipy import stats


# In[40]:


#exclude the row with z score more than 3
data[(np.abs(stats.zscore(data['z_score'])) < 3)]


# In[41]:


data.sort_index(inplace=True) # sort the data as per the index


# In[42]:


# Decompose the data frame to get the trend, seasonality and noise
plt.rcParams["figure.figsize"] = (15,8)
decompose_result = seasonal_decompose(data['Daily Visitors'],model='multiplicative',period=1)
decompose_result.plot()
plt.show()


# In[43]:


# Set the value of Alpha and define x as the time period
x = 12
alpha = 1/(2*x)


# In[44]:


# Single exponential smoothing of the visitors data set
data['HWES1'] = SimpleExpSmoothing(data['Daily Visitors']).fit(smoothing_level=alpha,optimized=False,use_brute=True).fittedvalues      
data[['Daily Visitors','HWES1']].plot(title='Holt Winters Single Exponential Smoothing grpah')


# In[45]:


# Double exponential smoothing of visitors data set ( Additive and multiplicative)
data['HWES2_ADD'] = ExponentialSmoothing(data['Daily Visitors'],trend='add').fit().fittedvalues
data['HWES2_MUL'] = ExponentialSmoothing(data['Daily Visitors'],trend='mul').fit().fittedvalues
data[['Daily Visitors','HWES2_ADD','HWES2_MUL']].plot(title='Holt Winters grapg: Additive Trend and Multiplicative Trend')


# In[46]:


data['HWES3_ADD'] = ExponentialSmoothing(data['Daily Visitors'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues

data['HWES3_MUL'] = ExponentialSmoothing(data['Daily Visitors'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues

data[['Daily Visitors','HWES3_ADD','HWES3_MUL']].plot(title='Holt Winters Triple Exponential Smoothing: Additive and Multiplicative Seasonality');


# In[47]:


# Split into train and test set
train_daily_visitors = dt[115:]
test_daily_visitors = dt[:115]


# In[48]:


test_daily_visitors


# In[49]:


train_daily_visitors


# In[50]:


# Fit the model
fitted_model = ExponentialSmoothing(train_daily_visitors['Daily Visitors'],trend='mul',seasonal='mul',seasonal_periods=2).fit()
test_predictions = fitted_model.forecast(58)
train_daily_visitors['Daily Visitors'].plot(legend=True,label='TRAIN')
test_daily_visitors['Daily Visitors'].plot(legend=True,label='TEST',figsize=(15,8))
test_predictions.plot(legend=True,label='PREDICTION')
plt.title('Train, Test and Predicted data points using Holt Winters Exponential Smoothing')


# In[51]:


fitted_model.summary()


# In[52]:


from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(train_daily_visitors["Daily Visitors"],test_predictions))
print('RMSE = '+str(rms))


# # **ARIMA**

# In[53]:


# Import libraries
from pandas import read_csv
from matplotlib import pyplot
from numpy import sqrt
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


# In[54]:


series = read_csv('C:\\Users\\admin\\Downloads\\EXECL R MATERIALS\\PROJECTS DETAILS\\Website Vistiors Daywise - Sheet1.csv', header=0, index_col=0, parse_dates=True)


# In[55]:


# separate out a validation dataset
split_point = len(series) - 10
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', header=False)
validation.to_csv('validation.csv', header=False)


# **Persistence/Base Model**

# In[56]:


# evaluate a persistence model
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
# load data
train = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
X = train.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]


# In[57]:


# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    yhat = history[-1]
    predictions.append(yhat)
# observation
    obs = test[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)


# #### ARIMA Hyperparameters

# In[58]:


# grid search ARIMA parameters for a time series

import warnings
from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt


# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
# prepare training dataset
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
# make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
# model_fit = model.fit(disp=0)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
# calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse


# #### Grid search for p,d,q values

# In[59]:


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float('inf'), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(train, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))


# In[60]:


# load dataset
train = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# evaluate parameters
p_values = range(0, 3)
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(train.values, p_values, d_values, q_values)


# #### Build Model based on the optimized values

# In[61]:


# save finalized model to file
from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA
import numpy


# In[62]:


# load data
train = read_csv('dataset.csv', header=0, index_col=0, parse_dates=True)
# prepare data
X = train.values
X = X.astype('float32')


# In[63]:


# fit model
model = ARIMA(X, order=(1,0,0))
model_fit = model.fit()
forecast=model_fit.forecast(steps=10)[0]
model_fit.plot_predict(1, 173)


# In[64]:


#Error on the test data
val=pd.read_csv('validation.csv',header=None)
rmse = sqrt(mean_squared_error(val[1], forecast))
rmse


# #### Combine train and test data and build final model

# In[65]:


# fit model
data = read_csv('C:\\Users\\admin\\Downloads\\EXECL R MATERIALS\\PROJECTS DETAILS\\Website Vistiors Daywise - Sheet1.csv', header=0, index_col=0, parse_dates=True)
# prepare data
X = train.values
X = X.astype('float32')


# In[66]:


model = ARIMA(X, order=(1,0,0))
model_fit = model.fit()


# In[67]:


forecast=model_fit.forecast(steps=162)[0]
model_fit.plot_predict(1,180)


# In[68]:


forecast


# In[69]:


# Actual vs Fitted
model_fit.plot_predict(dynamic=False)
plt.show()


# In[70]:


print(model_fit.summary())


# In[71]:


from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(train,forecast))
print('RMSE = '+str(rms))


# In[72]:


import pickle


# In[74]:


filename = "internet.sav"
pickle.dump(model, open(filename,"wb"))


# In[75]:


loaded_model = pickle.load(open("internet.sav", "rb"))


# In[ ]:




