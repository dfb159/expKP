import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as odr
from scipy import constants


# Read data:
pulse = np.loadtxt("calibration_period=0.64_range=10.24_working.txt", dtype=int, skiprows=12, usecols = 0)


# Initialize global variables:
channel = np.linspace(0, len(pulse)-1, num=len(pulse))


# Some functions to fit and calculate:
def Gaussian(parameter, x):
    A, mu, sigma, B = parameter
    return A * np.exp(-.5 * ((x - mu)/sigma)**2) + B

def line(parameter, x):
    m, n = parameter
    return m * x + n

def OneOrGreater(arg):
    if arg > 0.0:
        return np.sqrt(arg)
    else:
        return 1.0

#print(list(map(OneOrGreater, [25., 16., 4., 0.])))
#a, b = [12., 13.]
#print(a, b)

# Fit and analyze the peaks:
# Create a specified function, that returns an array:
def FitPeaksFromCalibrationSeparately(start, stop, A, mu, sigma, B):
    model = odr.Model(Gaussian)
    data = odr.RealData(channel[start:stop], pulse[start:stop], sx=np.repeat(0.5/np.sqrt(3), int(stop - start)), sy=list(map(OneOrGreater, pulse[start:stop])))
    ODR = odr.ODR(data, model, beta0=[A, mu, sigma, B])
    output = ODR.run()
    return [output.beta[1], output.sd_beta[1]]

# Calculate the positions of the peaks:
#print(FitPeaksFromCalibrationSeparately(0, 500, 1300., 275., 0.7, -0.02))
#print(FitPeaksFromCalibrationSeparately(300, 750, 1400., 516., 0.63, -0.04))
#print(FitPeaksFromCalibrationSeparately(550, 990, 1470., 757., 0.6, -0.02))
#print(FitPeaksFromCalibrationSeparately(800, 1200, 1600., 998., 0.6, -0.007))
#print(FitPeaksFromCalibrationSeparately(1100, 1400, 1600., 1239., 0.6, -0.05))


# Show results:
#plt.figure("Pulses", figsize=(10, 7.5))
#plt.plot(channel, pulse, ".k")
#plt.xlabel(u"Channel", fontsize=20)
#plt.ylabel(u"Pulse", fontsize=20)
#plt.xlim(0.0, 4000.)
#plt.grid(True)
#plt.savefig("Pulses.pdf")
#plt.show()