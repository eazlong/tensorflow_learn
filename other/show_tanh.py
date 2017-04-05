import numpy as np
import matplotlib.pyplot as plt
from numpy import arange
# http://stackoverflow.com/questions/7267226/range-for-floats
# 查看tanh
X=arange(-10,10,0.05)
print( X, X.shape )
y=np.tanh(X)
plt.scatter(X, y, c='r')
plt.show()
