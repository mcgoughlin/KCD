import numpy as np

a = np.array([0,1,2,3])
def assign(arr):
    arr[2] = 100
    return arr
b = a
print(a,b)
a = assign(a)
print(a,b)