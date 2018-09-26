import numpy as np


def getData(siz = 3):
    inp = np.array([[z/siz, 1] for z in range(1, siz)], dtype=float)
    out = np.array([[z/siz] for z in range(1, siz)], dtype=float)
    return inp, out


def getData1(siz = 3):
    inp = np.array(([0,0], [1,0], [0,1], [0,0]), dtype=float)
    out = np.array(([0], [1], [1], [0]), dtype=float)

    return inp, out



