import numpy as np

a = np.array([[1,2,3],[1,2,3],[1,2,3]])
print(a)
b = np.average(a,axis=0)
print(b)

a = np.array([1,2,3])
b = np.array([4,5,6])
c,d = np.meshgrid(a,b,indexing='ij')
print(c)