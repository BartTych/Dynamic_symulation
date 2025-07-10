import pickle
from matplotlib import pyplot as plt
import datetime


# Specify which results to plot

result = pickle.load(open('end_inf.pkl', 'rb'))

exc, end_x, end_y = zip(*result)
plt.scatter(end_x, end_y, s=2,c='b')
plt.show()
