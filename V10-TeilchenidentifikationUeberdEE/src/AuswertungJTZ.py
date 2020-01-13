import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as odr


# Read data:
countsE = np.loadtxt("histograms_adc_E_working.txt", dtype=int, usecols=1)
countsdE = np.loadtxt("histograms_adc_dE_working.txt", dtype=int, usecols=1)


# Initialize global variables:
n = len(countsE)
channel = np.linspace(0, n-1, num=n)


# Some functions to fit and calculate:
def Gaussian(parameter, x):
    A, mu, sigma, B = parameter
    return A * np.exp(-.5 * ((x - mu)/sigma)**2) + B

def line(parameter, x):
    m, n = parameter
    return m * x + n


# Apply a gaussian fit, to calculate the peak position, start at channel number 160:
model = odr.Model(Gaussian)
data = odr.RealData(, , sx=, sy=)
ODR = odr.ODR(data, model, beta0=[, , , ])
output = ODR.run()


# Show results:
plt.figure(figsize=(10, 7.5))
plt.plot(channel, countsE, ".k")
plt.xlabel(r"Channel $K$", fontsize=20)
plt.ylabel(r"Counts $N$", fontsize=20)
plt.grid(True)
plt.show()