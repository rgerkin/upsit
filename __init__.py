import csv
import numpy as np
import matplotlib.pyplot as plt

def cumul_hist(values,color='r'):
    sorted_values = sorted(values)
    quantiles = np.arange(0,1,1.0/len(values))
    plt.plot(sorted_values,quantiles,color)
    

