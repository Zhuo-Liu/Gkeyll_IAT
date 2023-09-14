import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

def solve():

    def func1(x):
        return x**(1/2) * np.sqrt(18.42 + np.log(x)) - 0.25*np.sqrt(np.pi/8)

    def func2(x):
        return x**(1/2) * np.sqrt(23 + np.log(x)) - 0.25*np.sqrt(np.pi/8)

    def to_E(x):
        return np.sqrt(x*x/25*2*0.02*0.02)

    x = np.arange(0.001,0.02,0.001)
    y1= func1(x)
    y2 = func2(x)

    plt.plot(x,y1,label='high W0')
    plt.plot(x,y2,label='low W0')
    plt.legend()
    plt.show()

    print(to_E(0.002))

solve()