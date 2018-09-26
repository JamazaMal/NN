import numpy as np
import TestData

x, y = TestData.getData()
print(x)

i = [[z, 1] for z in range(1, 4)]
print(i)

t = np.array([[z, 1] for z in range(1, 4)], dtype=float)
print(t)