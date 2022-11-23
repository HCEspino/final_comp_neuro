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

test = list(range(100))
p = [len(test) - i for i in test]
print(p)
print((softmax(p, 25) * 100).astype('float32'))