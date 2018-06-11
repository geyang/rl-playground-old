import os, dill
import matplotlib.pyplot as plt


def load_data():
    with open('./outputs/debug/data.pkl', 'rb') as f:
        while True:
            try:
                yield dill.load(f)
            except EOFError:
                return


data = [d['loss'] for d in load_data()]

fig = plt.figure()
plt.plot(data)
plt.show()
