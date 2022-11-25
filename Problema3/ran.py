import numpy as np
import pandas as pd
import random
import itertools
from itertools import permutations

def prepareArray(number, size):
    X1 = list(map(int, str(number)))
    while len(X1) < size:
        X1.insert(0, 0)
    return X1


columns = ["X1", "X2", "X3", "X4", "X5", "X6", "Y1", "Y2", "Y3", "Y4"]


def setFormat(first, second):
    total = first + second
    X1 = prepareArray(first, 3)
    X2 = prepareArray(second, 3)
    Y = prepareArray(total, 4)
    return X1 + X2 + Y


def createDataset(n, seed=42):
    max_number = 999
    np.random.seed(seed)
    random.seed(seed)
    values = np.linspace(0, max_number, 1000, dtype=int)
    unique_permutations = set(permutations(values, 2))
    samples = random.sample(unique_permutations, k=n)  # picks k number of numbers from 0 to 100 without replacement
    matrix = []
    for sample in samples:
        matrix.append(setFormat(sample[0], sample[1]))
    data = pd.DataFrame(matrix, columns=columns)
    data.drop_duplicates()
    return data


print(str(len(createDataset(25000, seed=42))))
