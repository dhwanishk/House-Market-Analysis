import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    
    # Getting training data (only first 30 rows)
    df = pd.read_csv('train.csv')
    x = df.drop('SalePrice', axis=1).iloc[:30, :].values
    y = df['SalePrice'].iloc[:30].values

    # for i in range(x.shape[1]):
    #     if np.issubdtype(x[:, i].dtype, np.str_):
    #         nan_mask = np.isnan(x[:, i])
    #         x[nan_mask, i] = 'nan'

    # print(x[0])    
    # nan_mask = np.isnan(x)
    # x[nan_mask] = 'nan'
    # print(x[:, 1].dtype)
    #print(x, y)
    
    
    # for i in range(len(df.columns)):
    #     n = i
    #     heading = df.columns[n]

    #     print(heading, x[:, n])


    #     plt.scatter(x[:, n], y, marker='x', c='r')
    #     plt.title(heading)
    #     plt.ylabel('Price (in 1000s of dollars)')
    #     plt.show()
    
main()