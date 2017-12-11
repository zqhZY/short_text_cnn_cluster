import numpy as np

def binarize(target):
    median = np.median(target, axis=1)[:, None]
    binary = np.zeros(shape=np.shape(target))
    binary[target > median] = 1
    return binary