#Run this to get the figure for the environment

import matplotlib.pyplot as plt
from env import softmax
import numpy as np

map_size_x = 10
map_size_y = 10

environment = np.array([[5, 5, 5, 5, 1, 1, 5, 5, 5, 5],
                        [5, 5, 5, 5, 1, 1, 5, 5, 5, 5],
                        [5, 5, 5, 15, 1, 1, 15, 5, 5, 5],
                        [5, 5, 15, 15, 1, 1, 15, 15, 5, 5],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [5, 5, 15, 15, 1, 1, 15, 15, 5, 5],
                        [5, 5, 5, 15, 1, 1, 15, 5, 5, 5],
                        [5, 5, 5, 5, 1, 1, 5, 5, 5, 5],
                        [5, 5, 5, 5, 1, 1, 5, 5, 5, 5]])

plt.imshow(environment)
plt.colorbar()
plt.xticks(np.arange(10))
plt.yticks(np.arange(10))
plt.title("Environment")
plt.show()