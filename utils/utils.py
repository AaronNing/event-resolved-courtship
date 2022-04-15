import numpy as np


def normalize(X: np.ndarray) -> np.ndarray:
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X


def find_consecutive(arr, min_consec):
    """
    eg: arr = [0,1,1,1,1,0,0,1,0,1,1,1], min_consec = 3
        out = [[1,5], 
               [9,12]]
    """
    i = 0
    consec = []
    for item in arr:
        if item:
            i += 1
        else:
            i = 0
        consec.append(i)

    for i in range(len(consec)-1):
        if consec[i+1] != 0 or consec[i] < min_consec:
            consec[i] = 0

    out = []
    for i, item in enumerate(consec):
        if item > 0:
            out.append([i - item + 1, i + 1])

    return np.array(out).astype(int)
    