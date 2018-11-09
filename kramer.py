import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

data = pd.read_csv("data.csv", header= None, skiprows = 1)

num = len(data.columns) # get data column
print(num)
# magnetic field for kramer plot
B1 = np.arange(8, 27)
out = B1

# constance for kramer plot
p = 0.5
q = 2

for i in range(1, num):
    df = data.loc[:, [0, i]]
    df = df.dropna()

    B = df[0] # magnetic field for data
    Jc = df[i] # Jc for data

    # convert dataframe to matrix
    B = B.as_matrix()
    Jc = Jc.as_matrix()
    BJc = (B**0.25)*(Jc**0.5) #for kramer plot

    # calucration of least square method

    A = np.array([B, np.ones(len(B))])
    A =A.T #transpose
    a, b = np.linalg.lstsq(A, BJc)[0]
    Bc2 = - b / a
    b1 = B1 / Bc2
    b2 = B / Bc2

    C = Jc / ((b2 ** (p - 1))*((1 - b2)**q))
    C = np.average(C)
    Jcfit = C * (b1**(p-1))*((1-b1)**q)
    
    ## plot
    out = np.c_[out, Jcfit]
    plt.plot(B1, Jcfit)
plt.show()

