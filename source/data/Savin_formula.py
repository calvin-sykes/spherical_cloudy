import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

def func(x, b): return np.exp(b/x)

t, a1, a2 = np.loadtxt("Savin2002_data.dat", unpack=True)
par, cov = curve_fit(func, t, a2/a1)

model = func(t, par)
modela = func(t, 42.5)

#plt.plot(np.log10(t), np.log10(a2/a1), 'bx')
#plt.plot(np.log10(t), np.log10(model), 'g-')
#plt.plot(np.log10(t), np.log10(modela), 'r-')
#plt.show()

fita1 = 2.0E-10 * (t**0.402) * np.exp(-37.1/t) - 3.31E-17 * (t**1.48)

#plt.plot(np.log10(t), np.log10(a2), 'bx')
plt.plot(np.log10(t), (np.exp(42.5/t)-a2/a1)/(a2/a1), 'b-')
plt.plot(np.log10(t), (np.exp(42.915/t)-a2/a1)/(a2/a1), 'g-')
#plt.plot(np.log10(t), np.log10(a1), 'rx')
#plt.plot(np.log10(t), np.log10(fita1), 'r-')
plt.show()