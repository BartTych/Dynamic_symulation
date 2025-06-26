import pickle
from matplotlib import pyplot as plt

result = pickle.load(open('result_multi.pkl', 'rb'))
for exc, end, step in result:
    plt.plot(step, end)
plt.show()
