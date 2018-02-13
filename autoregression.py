from pandas import Series
from matplotlib import pyplot
import pandas as pd
from pandas.plotting import lag_plot
from pandas import concat
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf



"""
Following this site to do prediction using Python: https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
"""


df = pd.read_csv('daily-minimum-temperatures.csv', header=0)
df = df.head(3600)

"""
1. plot the data
"""

try: 
    df['temperatures']=df['temperatures'].astype(float)
except:
    pass

# df.plot()


"""
2. Plot lag plot 
If sample lag plot of set exhibits a linear pattern, that means the data are strongly non-random and 
further suggests that an autoregressive model might be appropriate.
"""
# lag_plot(df['temperatures'], 1)


"""
3. Calculate auto-correlation -- correlation between t and t-1, absolute value will be 0-1, the stronger auto-correlated when close to 1; weaker when close to 0
"""
dataframe = concat([df.shift(1)['temperatures'], df['temperatures']], axis=1)
dataframe.columns = ['t-1', 't+1']
result = dataframe.corr()
print(result)


# autocorrelation_plot(df['temperatures'])
# plot_acf(df['temperatures'], lags=31)


"""
4. Predict based on itself, which is to say, use y(t) = y(t-1) as prediction. Use this one as a baseline
"""



from sklearn.metrics import mean_squared_error
# create lagged dataset
dataframe = concat([df['temperatures'].shift(1), df['temperatures']], axis=1)
dataframe.columns = ['t-1', 't+1']
# split into train and test sets
X = dataframe.values



train, test = X[1:len(X)-7], X[len(X)-7:]

train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
 
# persistence model
def model_persistence(x):
	return x
 
# walk-forward validation
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)
# plot predictions vs expected
pyplot.plot(test_y)
pyplot.plot(predictions, color='yellow')
pyplot.show()


"""
5. Use auto-regression to predict
"""
from statsmodels.tsa.ar_model import AR
# split dataset
X = df['temperatures'].values
train, test = X[1:len(X)-7], X[len(X)-7:]

# train autoregression
model = AR(train)
model_fit = model.fit()

print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)


# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='green')
pyplot.show()




