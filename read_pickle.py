import pickle
from matplotlib import pyplot as plt

exclog,end_log = pickle.load(open('exc_end_log.pkl', 'rb'))
plt.plot(end_log)
plt.show()
