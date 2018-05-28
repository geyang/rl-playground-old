import time

from main import dash
import numpy as np
from moleskin import moleskin as M

batch_size = 10

@M.timeit
def plot_demo(i):
    Y = np.array([[i, i+1, i+2]] * batch_size)
    X = np.ones((batch_size, 3)) * i + np.expand_dims(np.arange(batch_size), axis=1)

    dash.append('some loss', 'line', Y, X=X, opts=dict(legend=['red', 'green', 'yellow']))
    print(i, "works")
    # time.sleep(1)


for i in range(100):
    plot_demo(i)
