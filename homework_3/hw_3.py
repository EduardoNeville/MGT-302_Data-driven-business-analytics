## First we import the corresponding packages for the assignment
## We then shall download the data from the .xlsx file 

import numpy as np
import pandas as pd 
from scipy.stats import norm 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

#calculate single asset and portfolio returns 
def getReturns(df, ticker): 
    # Creating a new dataframe with the returns
    return df[ticker].pct_change()

def main():
    #read data from Excel
    #enter your own folder with the data file here
    data = pd.read_excel(r'data.xlsx') 
    df = pd.DataFrame(data, columns = ['NESN SW', 'ROG SW', 'NOVN SW', 'EURUSD', 'CHFUSD'])
    print(df.head())

    #calculate returns for each asset
    cols = []
    for col in df.columns:
        cols[col] = getReturns(df, col) 
    plt.plot(cols, axis= 0)
    plt.title('Returns')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend(cols)
    plt.show()

if "__name__" == "__main__":
    print("Running main()")
    main()
