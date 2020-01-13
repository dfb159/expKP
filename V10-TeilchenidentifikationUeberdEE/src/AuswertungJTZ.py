import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as odr


# Read data:
countsE = np.loadtxt("histograms_adc_E_working.txt", dtype=int, usecols=1)
countsE_ = np.loadtxt("histograms_adc_dE_working.txt", dtype=int, usecols=1)
countsdE = np.loadtxt("histograms_adc_dE_working.txt", dtype=int, usecols=2)


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

def OneOrGreaterReturnSqrt(arg):
    if arg > 0.0:
        return np.sqrt(arg)
    else:
        return 1.0


#------------------------------------------------------------------------------


# Apply a gaussian fit to the spectrum of the $E$ detector, to calculate the peak position, start at channel number 163:
model = odr.Model(Gaussian)
data = odr.RealData(channel[163:], countsE[163:], sx=np.repeat(0.5/np.sqrt(3), int(n - 163)), sy=list(map(OneOrGreaterReturnSqrt, countsE[163:])))
ODR = odr.ODR(data, model, beta0=[450., 170., 5., 0.05])
output = ODR.run()


# Apply the calibration fit in MeV:
channelCal = np.linspace(0., output.beta[1], 2)
channelCalUnc = np.linspace(0.5/np.sqrt(3), output.beta[2], 2)
energy = np.linspace(0., 5.48556, 2)
energyUnc = np.linspace(1., 0.0000012, 2)

model1 = odr.Model(line)
data1 = odr.RealData(channelCal, energy, sx=channelCalUnc, sy=energyUnc)
ODR1 = odr.ODR(data1, model1, beta0=[0.03, 0.])
output1 = ODR1.run()


#------------------------------------------------------------------------------


# Apply a gaussian fit to the spectrum of the $E$ detector, to calculate the position of the shifted peak:



#------------------------------------------------------------------------------


# Show results:
# Energy spectrum of the $E$ detector:
plt.figure(figsize=(10, 7.5))
plt.axvline(x=channel[163], color="C1", linestyle="--", linewidth=2.5, zorder=4)
plt.errorbar(channel, countsE, xerr=np.repeat(0.5/np.sqrt(3), n), yerr=list(map(OneOrGreaterReturnSqrt, countsE)), capsize=5., ecolor="C0", fmt=".C0", zorder=2, label="Messwerte")
plt.plot(channel, Gaussian(output.beta, channel), "-r", linewidth=3.5, zorder=3, label="Gaußsche Anpassungskurve")
plt.legend(loc="best", fontsize=18)
plt.xlabel(r"Kanal $K$", fontsize=20)
plt.ylabel(r"Counts $N$", fontsize=20)
plt.grid(True)
plt.show()
print(output.beta)
print(output.sd_beta)

# Calibration fit for the $E$ detector:
plt.figure(figsize=(10, 7.5))
plt.errorbar(channelCal, energy, xerr=channelCalUnc, yerr=np.linspace(0., 0.0000012, 2), capsize=5., ecolor="C0", fmt=".C0", zorder=2, label="Messwerte")
plt.plot(channel, line(output1.beta, channel), "-r", linewidth=3.5, zorder=3, label="Lineare Anpassungskurve")
plt.legend(loc="best", fontsize=18)
plt.xlabel(r"Kanal $K$", fontsize=20)
plt.ylabel(r"Energie $E$ $\left[ MeV\right]$", fontsize=20)
plt.grid(True)
plt.show()
print(output1.beta)
print(output1.sd_beta)

# Energy spectrum of the $E$ detector:
plt.figure(figsize=(10, 7.5))
plt.errorbar(channel, countsE_, xerr=np.repeat(0.5/np.sqrt(3), n), yerr=list(map(OneOrGreaterReturnSqrt, countsE_)), capsize=5., ecolor="C0", fmt=".C0", zorder=2, label="Messwerte")
plt.legend(loc="best", fontsize=18)
plt.xlabel(r"Kanal $K$", fontsize=20)
plt.ylabel(r"Counts $N$", fontsize=20)
plt.grid(True)
plt.show()

# Energy spectrum of the $\Delta E$ detector:
plt.figure(figsize=(10, 7.5))
plt.plot(channel, countsdE, ".k")
plt.grid(True)
plt.show()