import numpy as np


def getData(siz = 3):  # SIN
    r = np.random.rand(siz)
    inp = np.array([[z] for z in r], dtype=float)
    out = np.array([[np.sin(z)] for z in r], dtype=float)
    return inp, out


def getData_Root(siz = 3):  # Root
    r = np.random.rand(siz)
    inp = np.array([[z] for z in r], dtype=float)
    out = np.array([[z**0.5] for z in r], dtype=float)
    return inp, out


def getData_Line(siz = 3):  # straight line, with input bias
    inp = np.array([[z/siz, 1] for z in range(1, siz)], dtype=float)
    out = np.array([[z/siz] for z in range(1, siz)], dtype=float)
    return inp, out


def getData_XOR(siz = 3):  # XOR
    inp = np.array(([0,0], [1,0], [0,1], [0,0]), dtype=float)
    out = np.array(([0], [1], [1], [0]), dtype=float)

    return inp, out



