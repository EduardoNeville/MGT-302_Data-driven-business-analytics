r"""°°°
#### Homework 3 
#### Eduardo Neville & Åke Janson 
#### Spring 2023
°°°"""
# |%%--%%| <8ug8iJERAw|auHIgrYM8r>
r"""°°°
###### We import the packages to the jupyter notebook 
°°°"""
# |%%--%%| <auHIgrYM8r|hpkAUU8CtG>

import numpy as np
from scipy.stats import norm 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import pandas as pd


#|%%--%%| <hpkAUU8CtG|MYyXbziN3W>
r"""°°°
We then import the excel file and create a dataframe 
°°°"""
# |%%--%%| <MYyXbziN3W|uKs6tXPgkW>

#read the excel file
data = pd.read_excel(r'data.xlsx') 
df = pd.DataFrame(data, columns = ['NESN SW', 'ROG SW', 'NOVN SW', 'EURUSD', 'CHFUSD'])
print(df.head())
# show the growth of the assets
plt.plot(df)
plt.legend(df.columns)

#|%%--%%| <uKs6tXPgkW|XszYvdJwiJ>
r"""°°°
We then print the daily simple returns of each asset
°°°"""
#|%%--%%| <XszYvdJwiJ|ZAHVXm87Ms>

#calculate the daily simple returns
def percentage_change(df):
    returns = df.apply(lambda x: x.pct_change(1).dropna().reset_index(drop=True)) 
    return returns

#apply the function to the dataframe
plt.plot(percentage_change(df))
plt.legend(df.columns)
plt.show()

#|%%--%%| <ZAHVXm87Ms|9U1twcRPDN>

#calculate VaR and ES given the historical simulation method
def getVaR_ES(alpha, sorted_returns):
    index = int(np.ceil(alpha*(len(sorted_returns))))
    VaR = sorted_returns.apply(lambda x: -x[index])
    ES = sorted_returns.apply(lambda x: -x[:index].mean())
    return VaR, ES

#|%%--%%| <9U1twcRPDN|WeiZ6cYlam>

#calculate returns, sort them, and calculate the VaR and ES (historical simulation method)
def var_esHist(alpha, df): 
    returns = percentage_change(df) 
    sorted_returns = returns.apply(lambda x: x.sort_values().reset_index(drop=True)) 
    return getVaR_ES(alpha, sorted_returns)


#|%%--%%| <WeiZ6cYlam|SHYnG8Jp6f>

var_esHist(0.1,df), var_esHist(0.01,df), var_esHist(0.001,df)

#|%%--%%| <SHYnG8Jp6f|dRK6bs7d47>

#analytical computation of Value at Risk for a normally distributed random variable
def varNorm(mu, sigma, alpha):
    quantile = norm.ppf(alpha)
    return -mu - sigma*quantile

# |%%--%%| <dRK6bs7d47|t05FvEJKj8>

#analytical computation of Expected Shortfall for a normally distributed random variable
def esNorm(mu, sigma, alpha):
    quantile = norm.ppf(alpha)
    return -mu + sigma/alpha*norm.pdf(quantile)

# |%%--%%| <t05FvEJKj8|XZiFPcmFDN>
r"""°°°
# Monte Carlo Simulation
°°°"""
# |%%--%%| <XZiFPcmFDN|5MlVqSiMpR>

#Monte Carlo simulation method: Value at Risk for a normally distributed random variable
def varNormMC(mu, sigma, alpha, numSim):
    simulatedSample = np.random.normal(loc=mu, scale=sigma, size=numSim)
    return -np.quantile(simulatedSample, alpha)

# |%%--%%| <5MlVqSiMpR|UG4twI70Ub>

#Monte Carlo simulation method: Expected Shortfall for a normally distributed random variable
def esNormMC(mu, sigma, alpha, numSim):
    simulatedSample = np.random.normal(loc=mu, scale=sigma, size=numSim)
    sortedSample = np.sort(simulatedSample)
    index = int(np.ceil(numSim*alpha))
    return -np.average(sortedSample[0:index])

# |%%--%%| <UG4twI70Ub|RAPCQCe9qY>

mu, sigma, alpha = 0.04, 0.15, 0.01
varNorm(mu, sigma, alpha), esNorm(mu, sigma, alpha)

# |%%--%%| <RAPCQCe9qY|UY1zBTaq7H>

numSim = 1000
varNormMC(mu, sigma, alpha, numSim), esNormMC(mu, sigma, alpha, numSim)

#|%%--%%| <UY1zBTaq7H|vmz74hkNix>

#bootstrapped distribution of MC computed Value at Risk and Expected Shortfall for a Gaussian random variable
def bootstrapNormVaRES(mu, sigma, alpha, numSim, numBootstraps, numBins):
    varDistribution = np.empty(numBootstraps)
    esDistribution = np.empty(numBootstraps)
    for i in range(numBootstraps):
        varDistribution[i] = varNormMC(mu, sigma, alpha, numSim)
        esDistribution[i] = esNormMC(mu, sigma, alpha, numSim)
    plt.hist(varDistribution, numBins)
    plt.legend(['VaR'])
    plt.title("Value at Risk distribution with alpha {} and {} simulations".format(alpha, numSim))
    plt.show()
    plt.hist(esDistribution, numBins)
    plt.legend(['ES'])
    plt.title("Expected Shortfall distribution with alpha {} and {} simulations".format(alpha, numSim))
    plt.show()

# |%%--%%| <vmz74hkNix|Nl8nYWIrFs>

numSim = 100
numBootstraps = 1000
numBins = 30
alpha = 0.05
bootstrapNormVaRES(mu, sigma, alpha, numSim, numBootstraps, numBins)
numSim = 10000
bootstrapNormVaRES(mu, sigma, alpha, numSim, numBootstraps, numBins)

# |%%--%%| <Nl8nYWIrFs|bGQN0PylgC>

#calculate returns, fit a univariate normal distribution, and calculate VaR and ES given the variance-covariance method under Gaussianity
def var_esVarCovNorm(alpha, df): 
    returns = percentage_change(df) 
    moments = returns.apply(lambda x: norm.fit(x)) 
    return moments.apply(lambda x: varNorm(x[0], x[1], alpha)), moments.apply(lambda x: esNorm(x[0], x[1], alpha)) 

# |%%--%%| <bGQN0PylgC|h75XzLVT0k>

var_esVarCovNorm(0.1, df), var_esVarCovNorm(0.01, df), var_esVarCovNorm(0.001, df) 

# |%%--%%| <h75XzLVT0k|9MINCGPCgO>

#difference between the historical simulation and the variance-covariance (Gaussian-based) estimation
print(var_esHist(0.1,df)[0]-var_esVarCovNorm(0.1, df)[0])
print(var_esHist(0.1,df)[1]-var_esVarCovNorm(0.1, df)[1])
print(var_esHist(0.01,df)[0]-var_esVarCovNorm(0.01, df)[0]) 
print(var_esHist(0.01,df)[1]-var_esVarCovNorm(0.01, df)[1])
print(var_esHist(0.001,df)[0]-var_esVarCovNorm(0.001, df)[0])
print(var_esHist(0.001,df)[1]-var_esVarCovNorm(0.001, df)[1])




