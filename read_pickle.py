import pickle
from matplotlib import pyplot as plt

end_log = pickle.load(open('end_log.pkl', 'rb'))
plt.plot(end_log)
plt.show()
