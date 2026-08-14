import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set_theme(style='ticks')


x0 = 1.1
x1 = 2.3

def f(x):
    global x0
    return 1.2*(x-x0)

def phi(r):
    return np.where(np.abs(r)>1, 0., np.pow(1-np.abs(r), 2))

a = 2.1*abs(x0 - x1)

def h(x):
    global x0,x1,a
    return f(x) - f(x1)*phi(abs(x-x1)/a)


X = np.linspace(0,4,1000)
fY = f(X)
hY = h(X)

fig,ax = plt.subplots()
ax.plot(X,fY, linestyle='dashed', color="red", label="input function")
ax.plot(X,hY, label="corrected")
ax.grid(True, which='both', axis="both")
seaborn.despine(ax=ax, offset=0) # the important part here
fig.savefig("test.eps", format='eps', bbox_inches="tight")
plt.show()
