import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as odr
from scipy.interpolate import interp1d


# Read data:
countsE = np.loadtxt("histograms_adc_E_working.txt", dtype=int, usecols=1)
countsE_ = np.loadtxt("histograms_adc_dE_working.txt", dtype=int, usecols=1)
countsdE = np.loadtxt("histograms_adc_dE_working.txt", dtype=int, usecols=2)


# Initialize global variables:
n = len(countsE)
channel = np.linspace(0, n-1, num=n)
E_0 = 5485.56
dx = 1e-4
IonEnergy = np.array([1000., 1100., 1200., 1300., 1400., 1500., 1600., 1700., 1800., 2000., 2250., 2500., 2750., 3000., 3250., 3500., 3750., 4000., 4500., 5000., 5500., 6000., 6500., 7000.])
dEdxElec = np.array([304.6, 296.1, 287.8, 279.8, 272.2, 264.8, 257.9, 251.2, 245., 233.3, 220.4, 208.9, 198.8, 189.7, 181.5, 174.1, 167.4, 161.3, 150.5, 141.4, 133.4, 126.5, 120.3, 114.9])
dEdxNuclear = np.array([0.427, 0.3945, 0.367, 0.3433, 0.3227, 0.3046, 0.2885, 0.2742, 0.2613, 0.239, 0.2163, 0.1978, 0.1824, 0.1693, 0.1581, 0.1484, 0.1398, 0.1323, 0.1195, 0.1091, 0.1004, 0.09313, 0.08687, 0.08144])


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


# Apply a gaussian fit to the channel spectrum of the $E$ detector, to calculate the peak position, start at channel number 163:
model = odr.Model(Gaussian)
data = odr.RealData(channel[163:], countsE[163:], sx=np.repeat(0.5/np.sqrt(3), int(n - 163)), sy=list(map(OneOrGreaterReturnSqrt, countsE[163:])))
ODR = odr.ODR(data, model, beta0=[450., 170., 5., 0.05])
output = ODR.run()


# Apply the calibration fit for the $E$ detector in MeV:
EchannelCal = np.linspace(0., output.beta[1], 2)
EchannelCalUnc = np.linspace(0.5/np.sqrt(3), output.beta[2], 2)
EenergyCal = np.linspace(0., E_0, 2)
EenergyCalUnc = np.linspace(1., 0.0012, 2)

model1 = odr.Model(line)
data1 = odr.RealData(EchannelCal, EenergyCal, sx=EchannelCalUnc, sy=EenergyCalUnc)
ODR1 = odr.ODR(data1, model1, beta0=[32., 0.001])
output1 = ODR1.run()


#------------------------------------------------------------------------------


# Apply a gaussian fit to the spectrum of the $E$ detector, to calculate the position of the shifted peak, start at channel number 110, last used channel number is 135:
model2 = odr.Model(Gaussian)
data2 = odr.RealData(channel[110:136], countsE_[110:136], sx=np.repeat(0.5/np.sqrt(3), int(136 - 110)), sy=list(map(OneOrGreaterReturnSqrt, countsE_[110:136])))
ODR2 = odr.ODR(data2, model2, beta0=[248., 131., 1.95, 0.04])
output2 = ODR2.run()


# Calculate the energy loss of the particle in the $\Delta E$ detector by using channel numbers:
EnergyLoss = output1.beta[0] * (output.beta[1] - output2.beta[1])
EnergyLossUnc = output1.beta[0] * np.sqrt(output.beta[2]**2 + output2.beta[2]**2)


# Apply a gaussian fit to the spectrum of the $\Delta E$ detector with 75<=K<=95:
model3 = odr.Model(Gaussian)
data3 = odr.RealData(channel[75:96], countsdE[75:96], sx=np.repeat(0.5/np.sqrt(3), int(96 - 75)), sy=list(map(OneOrGreaterReturnSqrt, countsdE[75:96])))
ODR3 = odr.ODR(data3, model3, beta0=[242., 84., 4.5, 1.04])
output3 = ODR3.run()


# Apply the calibration fit for the $\Delta E$ detector in MeV:
dEchannelCal = np.linspace(0., output3.beta[1], 2)
dEchannelCalUnc = np.linspace(0.5/np.sqrt(3), output3.beta[2], 2)
dEenergyCal = np.linspace(0., EnergyLoss, 2)
dEenergyCalUnc = np.linspace(1., EnergyLossUnc, 2)

model4 = odr.Model(line)
data4 = odr.RealData(dEchannelCal, dEenergyCal, sx=dEchannelCalUnc, sy=dEenergyCalUnc)
ODR4 = odr.ODR(data4, model4, beta0=[14.5, 0.001])
output4 = ODR4.run()


# Interpolate the given data (IonEnergy and dEdx):
dEdx = dEdxElec + dEdxNuclear
dEdx_interpolated = interp1d(IonEnergy, dEdx, kind="cubic")


# Calculate the thickness of the $\Delta E$ detector by integration:
E = E_0
x = 0
while E >= E_0 - EnergyLoss:
    E = E - dEdx_interpolated(E) * dx
    x = x + dx


#------------------------------------------------------------------------------


# Show results:
# Channel spectrum of the $E$ detector:
plt.figure(figsize=(10, 7.5))
plt.axvline(x=channel[163 - 1], color="C1", linestyle="--", linewidth=2.5, zorder=4)
plt.errorbar(channel, countsE, xerr=np.repeat(0.5/np.sqrt(3), n), yerr=list(map(OneOrGreaterReturnSqrt, countsE)), capsize=5., ecolor="C0", fmt=".C0", zorder=2, label="Messwerte")
plt.plot(channel, Gaussian(output.beta, channel), "-r", linewidth=3.5, zorder=3, label="Gaußsche Anpassungskurve")
plt.legend(loc="best", fontsize=18)
plt.xlabel(r"Kanal $K$", fontsize=20)
plt.ylabel(r"Counts $N$", fontsize=20)
plt.tick_params(labelsize=16)
plt.grid(True)
plt.show()
print(output.beta)
print(output.sd_beta)

# Calibration fit for the $E$ detector:
plt.figure(figsize=(10, 7.5))
plt.errorbar(EchannelCal, EenergyCal, xerr=EchannelCalUnc, yerr=np.linspace(0., 0.0000012, 2), capsize=5., ecolor="C0", fmt=".C0", zorder=2, label="Messwerte")
plt.plot(channel, line(output1.beta, channel), "-r", linewidth=3.5, zorder=3, label="Lineare Anpassungskurve")
plt.legend(loc="best", fontsize=18)
plt.xlabel(r"Kanal $K$", fontsize=20)
plt.ylabel(r"Energie $E$ $\left[ keV\right]$", fontsize=20)
plt.tick_params(labelsize=16)
plt.grid(True)
plt.show()
print(output1.beta)
print(output1.sd_beta)


# Channel spectrum of the $E$ detector:
plt.figure(figsize=(10, 7.5))
plt.axvline(x=channel[109], color="C1", linestyle="--", linewidth=2.5, zorder=4)
plt.axvline(x=channel[136], color="C1", linestyle="--", linewidth=2.5, zorder=4)
plt.errorbar(channel, countsE_, xerr=np.repeat(0.5/np.sqrt(3), n), yerr=list(map(OneOrGreaterReturnSqrt, countsE_)), capsize=5., ecolor="C0", fmt=".C0", zorder=2, label="Messwerte")
plt.plot(channel, Gaussian(output2.beta, channel), "-r", linewidth=3.5, zorder=3, label="Gaußsche Anpassungskurve")
plt.legend(loc="best", fontsize=18)
plt.xlabel(r"Kanal $K$", fontsize=20)
plt.ylabel(r"Counts $N$", fontsize=20)
plt.tick_params(labelsize=16)
plt.grid(True)
plt.show()
print(output2.beta)
print(output2.sd_beta)

# Channel spectrum of the $\Delta E$ detector:
plt.figure(figsize=(10, 7.5))
plt.axvline(x=channel[74], color="C1", linestyle="--", linewidth=2.5, zorder=4)
plt.axvline(x=channel[96], color="C1", linestyle="--", linewidth=2.5, zorder=4)
plt.errorbar(channel, countsdE, xerr=np.repeat(0.5/np.sqrt(3), n), yerr=list(map(OneOrGreaterReturnSqrt, countsdE)), capsize=5., ecolor="C0", fmt=".C0", zorder=2, label="Messwerte")
plt.plot(channel, Gaussian(output3.beta, channel), "-r", linewidth=3.5, zorder=3, label="Gaußsche Anpassungskurve")
plt.legend(loc="best", fontsize=18)
plt.xlabel(r"Kanal $K$", fontsize=20)
plt.ylabel(r"Counts $N$", fontsize=20)
plt.tick_params(labelsize=16)
plt.grid(True)
plt.show()
print(output3.beta)
print(output3.sd_beta)

# Calibration fit for the $\Delta E$ detector:
plt.figure(figsize=(10, 7.5))
plt.errorbar(dEchannelCal, dEenergyCal, xerr=dEchannelCalUnc, yerr=np.linspace(0., EnergyLossUnc, 2), capsize=5., ecolor="C0", fmt=".C0", zorder=2, label="Messwerte")
plt.plot(channel, line(output4.beta, channel), "-r", linewidth=3.5, zorder=3, label="Lineare Anpassungskurve")
plt.legend(loc="best", fontsize=18)
plt.xlabel(r"Kanal $K$", fontsize=20)
plt.ylabel(r"Energie $E$ $\left[ keV\right]$", fontsize=20)
plt.tick_params(labelsize=16)
plt.grid(True)
plt.show()
print(output4.beta)
print(output4.sd_beta)

plt.figure(figsize=(10, 7.5))
plt.axvline(x=E_0, color="C1", linestyle="--", linewidth=2.5, zorder=4)
plt.axvline(x=E_0 - EnergyLoss, color="C1", linestyle="--", linewidth=2.5, zorder=4)
plt.plot(IonEnergy, dEdx, marker="o", markersize=9., color="C0", zorder=3, label="Tabellenwerte")
plt.plot(np.linspace(1000., 7000., 6000), dEdx_interpolated(np.linspace(1000., 7000., 6000)), "-r", linewidth=3.5, zorder=2, label="Kubische Spline-Interpolation")
plt.legend(loc="best", fontsize=18)
plt.xlabel(r"Energie $E$ des $\alpha$-Teilchens $\left[ keV\right]$", fontsize=20)
plt.ylabel(r"Energieverlust $\frac{dE}{dx}$ $\left[\frac{keV}{\mu m}\right]$", fontsize=20)
plt.xlim(3200., 6800.)
plt.ylim(100, 225.)
plt.xticks(np.linspace(3500., 6500., 7, endpoint=True))
plt.yticks(np.linspace(100., 225., 6, endpoint=True))
plt.tick_params(labelsize=16)
plt.grid(True)
plt.show()
print(E_0 - EnergyLoss)
print(E)
print(x)