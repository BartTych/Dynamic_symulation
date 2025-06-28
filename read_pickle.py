import pickle
from matplotlib import pyplot as plt
import datetime





plot_numbers = [0, 1, 2]  # Specify which results to plot
result = pickle.load(open('end.pkl', 'rb'))
for i, (exc, end, step) in enumerate(result):
    if i in plot_numbers:
        #plt.plot(step, exc, label='excitation')
        plt.plot(step, end, label='end response')

plt.show()
