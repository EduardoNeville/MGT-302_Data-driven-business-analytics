r"""°°°
# Homework 3 

#### EDUARDO NEVILLE
#### Åke Jason 
#### Spring 2023
°°°"""
# |%%--%%| <ep587P0Rao|WEywxXdf3w>
r"""°°°
    ## First we import the corresponding packages for the assignment
    ## We then shall download the data from the .xlsx file 
°°°"""
# |%%--%%| <WEywxXdf3w|UjgKPQnvTc>

import numpy as np
import pandas as pd 
from scipy.stats import norm 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt


# |%%--%%| <UjgKPQnvTc|bmVLc57JKg>

#read data from Excel
#enter your own folder with the data file here
data = pd.read_excel(r'data.xlsx') 
df = pd.DataFrame(data, columns = ['NESN SW', 'ROG SW', 'NOVN SW', 'EURUSD', 'CHFUSD'])
df.head()


#|%%--%%| <bmVLc57JKg|l9QbvEvSsg>

#calculate single asset and portfolio returns 
def getReturns(df): 
    # Creating a new dataframe with the returns
    returns = df.apply(lambda x: x.pct_change(1).dropna().reset_index(drop=True)) 
    returns['Portfolio CHF'] = returns['NESN SW']/3 + returns['ROG SW']/3 + returns['NOVN SW']/3 
    returns['Portfolio USD'] = returns['Portfolio CHF'] + returns['CHFUSD'] + returns['Portfolio CHF']*returns['CHFUSD'] 
    return returns 

plt.plot(df)


# |%%--%%| <l9QbvEvSsg|Q2xjnU6eVZ>

#simulate the data set

#generate a matrix of size 100 x 10 of standard normally distributed numbers
X = np.random.standard_normal(size=(n_simul, n_pred))

#generate a vector of size 100 x 1 of standard normally distributed numbers
Y = np.random.standard_normal(size=(n_simul, 1))

# split the data into training and test set based on the input above
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size)

# |%%--%%| <Q2xjnU6eVZ|eonAzhsTfV>

#initialize the mse_train variable
mse_train = np.zeros((n_pred + 1, 1))

#input the empty model and train the regression on a vector of zeros and as target variable Y train
reg = LinearRegression().fit(np.zeros((len(Y_train), 1)), Y_train)

#predict on the train set again with a vector of zeros
pred = reg.predict(np.zeros((len(Y_train), 1)))

#calculate the mean squared error of the prediction and Y train
mse_train[0] = mse(pred, Y_train)

#repeat the above procedure with an increasing number of predictors
for i in range(1, n_pred + 1):
    reg = LinearRegression().fit(X_train[:, :i], Y_train)
    pred = reg.predict(X_train[:, :i])
    mse_train[i] = mse(pred, Y_train)

# |%%--%%| <eonAzhsTfV|0NFqhAIPb0>

#initialize the mse_test variable
mse_test = np.zeros((n_pred + 1, 1))

#evaluate the empty model
reg = LinearRegression().fit(np.zeros((len(Y_train), 1)), Y_train)

#now predict on the test set
pred = reg.predict(np.zeros((len(Y_train), 1)))

#evaluate the prediction on the test set
mse_test[0] = mse(pred, Y_test)

#train the model on the train set and evaluate prediction on the test set
for i in range(1, n_pred + 1):
    reg = LinearRegression().fit(X_train[:, :i], Y_train) 
    pred = reg.predict(X_test[:, :i])
    mse_test[i] = mse(pred, Y_test)

# |%%--%%| <0NFqhAIPb0|dwjDwaBZUy>

#plot the train and test mean squared errors (MSEs)
plt.plot(mse_train, 'r--', label='Train')
plt.plot(mse_test, 'g--', label='Test')
plt.xlabel('Number of regressors')
plt.ylabel('MSE')
plt.legend(loc='upper left')
plt.show()

# |%%--%%| <dwjDwaBZUy|IY39sRbsNf>

#set the array of penalization parameters on which the two models will be tested
alphas = np.logspace(-4, 5, 100, endpoint=True, base=10.0)

# |%%--%%| <IY39sRbsNf|TIi8j813sv>

#shrinkage with the Ridge regression

#use the function validation curve to produce the mse on the all the training folds and the single validation fold
train_scores_ridge, test_scores_ridge = validation_curve(Ridge(), X, Y, scoring = 'neg_mean_squared_error', param_name = 'alpha', param_range = alphas, cv = K_Folds)

#train another ridge cross validated model to access the best alpha
ridge = RidgeCV(alphas = alphas, cv = K_Folds, fit_intercept=False).fit(X, np.ravel(Y))

#this is the alpha chosen by python whcih works best for the model
ridge_alpha = ridge.alpha_

#calculate the mean of the mse scores on the initially specified number of folds
train_scores_mean_ridge = np.mean(train_scores_ridge, axis=1)
test_scores_mean_ridge = np.mean(test_scores_ridge, axis=1)

# |%%--%%| <TIi8j813sv|B6EojUuhuW>

#shrinkage with the Lasso regression

#use the function validation curve to produce the mse on the all the training folds and the single validation fold
train_scores_lasso, test_scores_lasso = validation_curve(Lasso(), X, Y, scoring = 'neg_mean_squared_error', param_name = 'alpha', param_range = alphas, cv = K_Folds)

#train another ridge cross validated model to access the best alpha
#lasso = LassoCV(alphas = alphas, cv = K_Folds).fit(X, Y)
lasso = LassoCV(alphas = alphas, cv = K_Folds, fit_intercept=False).fit(X, np.ravel(Y))

#this is the alpha chosen by python whcih works best for the model
lasso_alpha = lasso.alpha_

#calculate the mean of the mse scores on the initially specified number of folds
train_scores_mean_lasso = np.mean(train_scores_lasso, axis=1)
test_scores_mean_lasso = np.mean(test_scores_lasso, axis=1)

# |%%--%%| <B6EojUuhuW|zQPcYnq15a>

#plot the cross-validated MSE curves for ridge and lasso
plt.plot(alphas, test_scores_mean_ridge, 'r--', label='Ridge')
plt.plot(alphas, test_scores_mean_lasso, 'g--', label='Lasso')
plt.xscale('log')
plt.xlabel('Penalization parameter')
plt.ylabel('MSE')
plt.legend(loc='upper left')
plt.show()
ridge_alpha, lasso_alpha

# |%%--%%| <zQPcYnq15a|bJgWLq6myM>

#initialize the ridge_coefs variable
ridge_coefs = []

#save the value of the ridge regression coefficients for each penalization parameter alpha
for a in alphas:
    ridge = Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, Y)
    ridge_coefs.append(np.ravel(ridge.coef_))

# |%%--%%| <bJgWLq6myM|nbDWWqleMy>

#initialize the lasso_coefs variable
lasso_coefs = []

#save the value of the lasso regression coefficients for each penalization parameter alpha
for a in alphas:
    lasso = Lasso(alpha=a, fit_intercept=False)
    lasso.fit(X, Y)
    lasso_coefs.append(np.ravel(lasso.coef_))

# |%%--%%| <nbDWWqleMy|PdtiMbdjbZ>

#plot the ridge regression coefficients in dependence of the penalization parameter alpha
ax = plt.gca()
ax.plot(alphas, ridge_coefs)
ax.set_xscale("log")
plt.xlabel('Penalization parameter')
plt.ylabel('Coefficients')
plt.title('Ridge coefficients as a function of regularization')
plt.show()

# |%%--%%| <PdtiMbdjbZ|IjgC5vnXHj>

#plot the lasso regression coefficients in dependence of the penalization parameter alpha
ax = plt.gca()
ax.plot(alphas, lasso_coefs)
ax.set_xscale("log")
plt.xlabel('Penalization parameter')
plt.ylabel('Coefficients')
plt.title('Lasso coefficients as a function of regularization')
plt.show()

# |%%--%%| <IjgC5vnXHj|mopB7QcL20>


