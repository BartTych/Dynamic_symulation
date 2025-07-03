import pickle
from matplotlib import pyplot as plt
import datetime


  # Specify which results to plot
result_to_plot = [0,1]
result = pickle.load(open('end.pkl', 'rb'))
for i, (exc, end, step) in enumerate(result):
            #plt.plot(step, exc, label='excitation')
    if i in result_to_plot:
      plt.plot(step, end, label='end response')
      #plt.plot(step, exc)

plt.show()
