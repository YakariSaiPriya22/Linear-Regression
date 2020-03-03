
import os
print(os.getcwd())

os.chdir("path")

#Step1 : Clear Business Understanding 
# Step 2: Data Gathering from different sources - 
#create a single data set

# Step 3 : Get the data into working environment
#Step3.1: Import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Step 3.2 : Read data from source
# Best Practice to set the working directory before we
# process the data
data = pd.read_csv("Data\GermanCredit.csv")

# Step 4: Sanity check
data.head()
sanitychecks = data.describe().T
data.describe().T.to_csv("output\\sanitychecks.csv")

# 1. shape of the data
# 2. Mising values
#   from describe or is.null.....
# 3. Missing imputations
# 4. Outlier checking
#   from box plots, graphs, from descriptive stats


#data distribution plot
#sns.pairplot(df)
sns.distplot(data['Amount'])

# Correlation
sns.heatmap(data.corr())# to run the corelation


# Step 5: EDA
# To gain more understanding about data
# lot of cross tabs, graphs, charts....

# Step 5: Feature engineering & Feature selection


# Step 6: Sampling
# splitting X and y into training and testing sets
list(data)
np.shape(data)
y= data['Amount']
list(y)
np.shape(y)
X = data.drop(['Amount','Class'], axis=1)
list(X) 
np.shape(X)

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=1)

np.shape(X_train)
np.shape(X_test)
np.shape(y_test)
np.shape(y_train)

# Step 7: Model Building - Train set
#Creating and Training the Model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit(X_train,y_train)

#Predictions from our Model
#Let's grab predictions off our Train set and see how well it did!
y_pred=lm.predict(X_train)

# Evaluate model
#calculate R2 value
from sklearn.metrics import r2_score

r2_score(y_train,y_pred)
print(round(r2_score(y_train,y_pred)*100,2),'%')

#Calculate root-mean-square error (RMSE)
import numpy as np
from sklearn.metrics import mean_squared_error

lin_mse = mean_squared_error(y_pred, y_train)
lin_rmse = np.sqrt(lin_mse)
print('Linear Regression RMSE: %.4f' % lin_rmse)



#Predictions from our Model
#Let's grab predictions off our test set and see how well it did!
y_pred_t=lm.predict(X_test)

#calculate R2 value
from sklearn.metrics import r2_score

r2_score(y_test,y_pred_t)
print(round(r2_score(y_test,y_pred_t)*100,2),'%')

#Calculate root-mean-square error (RMSE)
import numpy as np
from sklearn.metrics import mean_squared_error

lin_mse = mean_squared_error(y_pred_t, y_test)
lin_rmse = np.sqrt(lin_mse)
print('Linear Regression RMSE: %.4f' % lin_rmse)

# print the intercept and co-efficients
print(lm.intercept_)
print(lm.coef_)


# plot for residual error 

## setting plot style 
plt.style.use('fivethirtyeight') 

## plotting residual errors in training data 
plt.scatter(lm.predict(X_train), lm.predict(X_train) - y_train, 
			color = "green", s = 10, label = 'Train data') 

## plotting residual errors in test data 
plt.scatter(lm.predict(X_test), lm.predict(X_test) - y_test, 
			color = "blue", s = 10, label = 'Test data') 

## plotting line for zero residual error 
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2) 

## plotting legend 
plt.legend(loc = 'upper right') 

## plot title 
plt.title("Residual errors") 

## function to show plot 
plt.savefig("output\\image1.jpeg")
plt.show() 




list(data)

#Linearity (Y=b0+b1x1+b2x2+..+bnxn+E)
#Constant Variance  (Homoscadesticity = X incr ~ E incr)
#Independent Error Terms (Auto correlation, Snacky/curvy plot)
#Normal Errors (bell Shaped curve/NQQ)
#No Multi-collinearity (X Variables are indep)

# =============================================================================
# #Linearity - Residual Plot
# =============================================================================
#Normal Quantile-Quantile plot
import numpy as np 
import pylab 
import scipy.stats as stats

stats.probplot(resid, dist="norm", plot=pylab)
pylab.show()


# =============================================================================
# #Constant Variance Variance (Homoscadesticity)
# =============================================================================
#using residual plot
import seaborn
import matplotlib.pyplot as plt
w = 6; h = 4; d = 70
plt.figure(figsize=(w, h), dpi=d)
seaborn.residplot(lm.predict(X_train), lm.predict(X_train) - y_train)
plt.savefig("out.png")

resid=y_pred-y_train
resid=pd.Series(resid)
plt.scatter(y_pred,resid)
plt.hlines(0,0,12000)

plt.scatter(y_pred,y_pred)

# =============================================================================
# #Independance of errors (Auto Correlation)
# =============================================================================
plt.scatter(resid.index, resid.values)
plt.hlines(0,0,1000)

from statsmodels.stats.stattools import durbin_watson
durbin_watson(resid)

# =============================================================================
# #Normal Errors (bell Shaped curve/NQQ)
# =============================================================================
#Histogram
plt.hist(resid, bins='auto', color='green', alpha=0.7,rwidth=0.85)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title(' Histogram')


# =============================================================================
# #No Multi-collinearity (X Variables are indep -  VIF)
# =============================================================================
from sklearn.metrics import r2_score
#VIF = 1/1-r2
VIF = 1/ (1-r2_score(y_train,y_pred))
print(VIF)

#from statsmodels.stats.outliers_influence import variance_inflation_factor
#vif = pd.DataFrame()
#vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
#vif["features"] = X.columns


# =============================================================================
# AIC and BIC
# =============================================================================
import math

#AIC= 2k - 2ln(sse) (k= # of variables)
sse = sum(resid**2)
AIC= 2*60 - 2*(math.log(sse))
print(AIC)

#BIC = n*ln(sse/n) + k*ln(n) (k= # of variables,n = # of observations)
sse = sum(resid**2)
BIC = 600*math.log(sse/600)+60*math.log(600)
print(BIC)


