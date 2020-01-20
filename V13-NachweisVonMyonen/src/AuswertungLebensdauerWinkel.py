import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as odr
#from scipy import constants


# Read data:
pulse = np.loadtxt("calibration_period=0.64_range=10.24_working.txt", dtype=int, skiprows=12, usecols = 0)
counts = np.loadtxt("Lebensdauer_working.txt", dtype=int, skiprows=12, usecols = 0)
Counts0Deg = np.loadtxt("landau_0_new_working.txt", dtype=int, skiprows=12, usecols = 0)
Counts30Deg = np.loadtxt("landau_30_working.txt", dtype=int, skiprows=12, usecols = 0) 
Counts45Deg = np.loadtxt("landau_45_working.txt", dtype=int, skiprows=12, usecols = 0)
Counts75Deg = np.loadtxt("landau_75_working.txt", dtype=int, skiprows=12, usecols = 0)


# Initialize global variables:
n = len(pulse)
channel = np.linspace(0, n-1, num=n)
ActiveMeasurementTime = np.array([78561., 88443., 91368., 260489.])
Angle = np.array([0., 30., 45., 75.])


# Some functions to fit and calculate:
def Gaussian(parameter, x):
    A, mu, sigma, B = parameter
    return A * np.exp(-.5 * ((x - mu)/sigma)**2) + B

def line(parameter, x):
    m, n = parameter
    return m * x + n

def lineUnc(m, m_err, n_err, x):
    return np.sqrt((m**2)/12. + (x * m_err)**2 + n_err**2)

def exponential(parameter, x):
    N0, tau, y0 = parameter
    return N0 * np.exp((-1. * x)/tau) + y0

def LandauDistribution(parameter, x):
    A, x0, b = parameter
    return (A * np.exp( -.5 * ( (x-x0)/b + np.exp(-1.*(x-x0)/b) ) ))/np.sqrt(2.*np.pi)

def KosinusQuadrat(A, theta):
    return A * (np.cos((np.pi * theta)/180.))**2

def OneOrGreater(arg):
    if arg > 0.0:
        return arg
    else:
        return 1.0

def OneOrGreaterReturnSqrt(arg):
    if arg > 0.0:
        return np.sqrt(arg)
    else:
        return 1.0


# Fit and analyze the peaks:
# Create a specified function, that returns an array:
def FitPeaksFromCalibrationSeparately(start, stop, A, mu, sigma, B):
    model = odr.Model(Gaussian)
    data = odr.RealData(channel[start:stop], pulse[start:stop], sx=np.repeat(0.5/np.sqrt(3), int(stop - start)), sy=list(map(OneOrGreaterReturnSqrt, pulse[start:stop])))
    ODR = odr.ODR(data, model, beta0=[A, mu, sigma, B])
    output = ODR.run()
    return [output.beta[1], output.sd_beta[1]]

# Calculate the positions of the peaks:
PeakPositions = np.zeros(15)
PeakPositionsUnc = np.zeros(15)
PeakPositions[0], PeakPositionsUnc[0] = FitPeaksFromCalibrationSeparately(0, 500, 1300., 275., 0.7, -0.02)
PeakPositions[1], PeakPositionsUnc[1] = FitPeaksFromCalibrationSeparately(300, 750, 1400., 516., 0.63, -0.04)
PeakPositions[2], PeakPositionsUnc[2] = FitPeaksFromCalibrationSeparately(550, 990, 1470., 757., 0.6, -0.02)
PeakPositions[3], PeakPositionsUnc[3] = FitPeaksFromCalibrationSeparately(800, 1200, 1600., 998., 0.6, -0.007)
PeakPositions[4], PeakPositionsUnc[4] = FitPeaksFromCalibrationSeparately(1100, 1400, 1600., 1239., 0.6, -0.05)
PeakPositions[5], PeakPositionsUnc[5] = FitPeaksFromCalibrationSeparately(1300, 1700, 1650., 1480., 0.57, -0.005)
PeakPositions[6], PeakPositionsUnc[6] = FitPeaksFromCalibrationSeparately(1500, 1900, 1660., 1721., 0.6, -0.005)
PeakPositions[7], PeakPositionsUnc[7] = FitPeaksFromCalibrationSeparately(1800, 2200, 1760., 1962.5, 0.6, -0.001)
PeakPositions[8], PeakPositionsUnc[8] = FitPeaksFromCalibrationSeparately(2000, 2400, 1650., 2203.5, 0.6, -0.001)
PeakPositions[9], PeakPositionsUnc[9] = FitPeaksFromCalibrationSeparately(2300, 2600, 1480., 2444.5, 0.63, -0.001)
PeakPositions[10], PeakPositionsUnc[10] = FitPeaksFromCalibrationSeparately(2500, 2900, 1330., 2685.5, 0.65, 0.005)
PeakPositions[11], PeakPositionsUnc[11] = FitPeaksFromCalibrationSeparately(2700, 3100, 1330., 2927., 0.7, 0.005)
PeakPositions[12], PeakPositionsUnc[12] = FitPeaksFromCalibrationSeparately(3000, 3400, 1270., 3168., 0.7, -0.001)
PeakPositions[13], PeakPositionsUnc[13] = FitPeaksFromCalibrationSeparately(3200, 3600, 1400., 3409.5, 0.7, -0.001)
PeakPositions[14], PeakPositionsUnc[14] = FitPeaksFromCalibrationSeparately(3500, 4000, 1390., 3650.5, 0.75, 0.001)


# Calculate the time intervals:
dt = np.zeros(15)
for i in range(0, 15):
    dt[i] = 0.64 * i


# Create the linear fit for the time calibration:
model0 = odr.Model(line)
data0 = odr.RealData(PeakPositions, dt, sx=PeakPositionsUnc)
ODR0 = odr.ODR(data0, model0, beta0=[0.00267, -0.7])
output0 = ODR0.run()


# Use the time calibration to create the x-coordinates:
Time = line(output0.beta, channel)
TimeUnc = lineUnc(output0.beta[0], output0.sd_beta[0], output0.sd_beta[1], channel)


# Bin the channels and their counts:
BinnedCounts = np.zeros(n)
BinnedCountsUnc = np.zeros(n)
for i in range(0, n, 8):
    BinnedCounts[i] = .125 * (counts[i] + counts[i+1] + counts[i+2] + counts[i+3] + counts[i+4] + counts[i+5] + counts[i+6] + counts[i+7])
    BinnedCounts[i+1] = BinnedCounts[i]
    BinnedCounts[i+2] = BinnedCounts[i]
    BinnedCounts[i+3] = BinnedCounts[i]
    BinnedCounts[i+4] = BinnedCounts[i]
    BinnedCounts[i+5] = BinnedCounts[i]
    BinnedCounts[i+6] = BinnedCounts[i]
    BinnedCounts[i+7] = BinnedCounts[i]
    BinnedCountsUnc[i] = .125 * np.sqrt(counts[i] + counts[i+1] + counts[i+2] + counts[i+3] + counts[i+4] + counts[i+5] + counts[i+6] + counts[i+7])
    BinnedCountsUnc[i+1] = BinnedCountsUnc[i]
    BinnedCountsUnc[i+2] = BinnedCountsUnc[i]
    BinnedCountsUnc[i+3] = BinnedCountsUnc[i]
    BinnedCountsUnc[i+4] = BinnedCountsUnc[i]
    BinnedCountsUnc[i+5] = BinnedCountsUnc[i]
    BinnedCountsUnc[i+6] = BinnedCountsUnc[i]
    BinnedCountsUnc[i+7] = BinnedCountsUnc[i]


# Create an exponential fit for the lifetime and decay measurement:
# In order to ignore the first/last data points, start at the array element 136 and end at the array element 8008:
model1 = odr.Model(exponential)
data1 = odr.RealData(Time[136:8008], BinnedCounts[136:8008], sx=TimeUnc[136:8008], sy=list(map(OneOrGreater, BinnedCountsUnc[136:8008])))
ODR1 = odr.ODR(data1, model1, beta0=[26., 2.1, 0.7])
output1 = ODR1.run()


#------------------------------------------------------------------------------
# Energy loss and Landau distribution:
# For the 0 degrees spectrum:
# Bin the channels and their counts:
BinnedCounts0Deg = np.zeros(n)
BinnedCounts0DegUnc = np.zeros(n)
for i in range(0, n, 4):
    BinnedCounts0Deg[i] = .25 * (Counts0Deg[i] + Counts0Deg[i+1] + Counts0Deg[i+2] + Counts0Deg[i+3])
    BinnedCounts0Deg[i+1] = BinnedCounts0Deg[i]
    BinnedCounts0Deg[i+2] = BinnedCounts0Deg[i]
    BinnedCounts0Deg[i+3] = BinnedCounts0Deg[i]
    BinnedCounts0DegUnc[i] = .25 * np.sqrt(Counts0Deg[i] + Counts0Deg[i+1] + Counts0Deg[i+2] + Counts0Deg[i+3])
    BinnedCounts0DegUnc[i+1] = BinnedCounts0DegUnc[i]
    BinnedCounts0DegUnc[i+2] = BinnedCounts0DegUnc[i]
    BinnedCounts0DegUnc[i+3] = BinnedCounts0DegUnc[i]

# Create a Landau fit for the energy loss spectrum; because of background start at the array element 400:
model2 = odr.Model(LandauDistribution)
data2 = odr.RealData(channel[400:], BinnedCounts0Deg[400:]/ActiveMeasurementTime[0], sx=np.repeat(0.5/np.sqrt(3), int(n - 400)), sy=np.divide(list(map(OneOrGreater, BinnedCounts0DegUnc)), ActiveMeasurementTime[0])[400:])
ODR2 = odr.ODR(data2, model2, beta0=[0.002, 500., 250.])
output2 = ODR2.run()


# For the 30 degrees spectrum:
# Bin the channels and their counts:
BinnedCounts30Deg = np.zeros(n)
BinnedCounts30DegUnc = np.zeros(n)
for i in range(0, n, 4):
    BinnedCounts30Deg[i] = .25 * (Counts30Deg[i] + Counts30Deg[i+1] + Counts30Deg[i+2] + Counts30Deg[i+3])
    BinnedCounts30Deg[i+1] = BinnedCounts30Deg[i]
    BinnedCounts30Deg[i+2] = BinnedCounts30Deg[i]
    BinnedCounts30Deg[i+3] = BinnedCounts30Deg[i]
    BinnedCounts30DegUnc[i] = .25 * np.sqrt(Counts30Deg[i] + Counts30Deg[i+1] + Counts30Deg[i+2] + Counts30Deg[i+3])
    BinnedCounts30DegUnc[i+1] = BinnedCounts30DegUnc[i]
    BinnedCounts30DegUnc[i+2] = BinnedCounts30DegUnc[i]
    BinnedCounts30DegUnc[i+3] = BinnedCounts30DegUnc[i]

# Create a Landau fit for the energy loss spectrum; because of background start at the array element 400:
model3 = odr.Model(LandauDistribution)
data3 = odr.RealData(channel[400:], BinnedCounts30Deg[400:]/ActiveMeasurementTime[1], sx=np.repeat(0.5/np.sqrt(3), int(n - 400)), sy=np.divide(list(map(OneOrGreater, BinnedCounts30DegUnc)), ActiveMeasurementTime[1])[400:])
ODR3 = odr.ODR(data3, model3, beta0=[0.01, 500., 50.])
output3 = ODR3.run()


# For the 45 degrees spectrum:
# Bin the channels and their counts:
BinnedCounts45Deg = np.zeros(n)
BinnedCounts45DegUnc = np.zeros(n)
for i in range(0, n, 4):
    BinnedCounts45Deg[i] = .25 * (Counts45Deg[i] + Counts45Deg[i+1] + Counts45Deg[i+2] + Counts45Deg[i+3])
    BinnedCounts45Deg[i+1] = BinnedCounts45Deg[i]
    BinnedCounts45Deg[i+2] = BinnedCounts45Deg[i]
    BinnedCounts45Deg[i+3] = BinnedCounts45Deg[i]
    BinnedCounts45DegUnc[i] = .25 * np.sqrt(Counts45Deg[i] + Counts45Deg[i+1] + Counts45Deg[i+2] + Counts45Deg[i+3])
    BinnedCounts45DegUnc[i+1] = BinnedCounts45DegUnc[i]
    BinnedCounts45DegUnc[i+2] = BinnedCounts45DegUnc[i]
    BinnedCounts45DegUnc[i+3] = BinnedCounts45DegUnc[i]

# Create a Landau fit for the energy loss spectrum; because of background start at the array element 400:
model4 = odr.Model(LandauDistribution)
data4 = odr.RealData(channel[400:], BinnedCounts45Deg[400:]/ActiveMeasurementTime[2], sx=np.repeat(0.5/np.sqrt(3), int(n - 400)), sy=np.divide(list(map(OneOrGreater, BinnedCounts45DegUnc)), ActiveMeasurementTime[2])[400:])
ODR4 = odr.ODR(data4, model4, beta0=[0.01, 500., 50.])
output4 = ODR4.run()


# For the 75 degrees spectrum:
# Bin the channels and their counts:
BinnedCounts75Deg = np.zeros(n)
BinnedCounts75DegUnc = np.zeros(n)
for i in range(0, n, 4):
    BinnedCounts75Deg[i] = .25 * (Counts75Deg[i] + Counts75Deg[i+1] + Counts75Deg[i+2] + Counts75Deg[i+3])
    BinnedCounts75Deg[i+1] = BinnedCounts75Deg[i]
    BinnedCounts75Deg[i+2] = BinnedCounts75Deg[i]
    BinnedCounts75Deg[i+3] = BinnedCounts75Deg[i]
    BinnedCounts75DegUnc[i] = .25 * np.sqrt(Counts75Deg[i] + Counts75Deg[i+1] + Counts75Deg[i+2] + Counts75Deg[i+3])
    BinnedCounts75DegUnc[i+1] = BinnedCounts75DegUnc[i]
    BinnedCounts75DegUnc[i+2] = BinnedCounts75DegUnc[i]
    BinnedCounts75DegUnc[i+3] = BinnedCounts75DegUnc[i]

# Create a Landau fit for the energy loss spectrum; because of background start at the array element 450:
model5 = odr.Model(LandauDistribution)
data5 = odr.RealData(channel[450:800], BinnedCounts75Deg[450:800]/ActiveMeasurementTime[3], sx=np.repeat(0.5/np.sqrt(3), int(800 - 450)), sy=np.divide(list(map(OneOrGreater, BinnedCounts75DegUnc)), ActiveMeasurementTime[3])[450:800])
ODR5 = odr.ODR(data5, model5, beta0=[0.001, 500., 55.])
output5 = ODR5.run()


# Calculate the count rates and their uncertainties:
CountRates = np.divide(np.array([ sum(Counts0Deg[int(output2.beta[1] - output2.beta[2]):int(output2.beta[1] + output2.beta[2])]), sum(Counts30Deg[int(output3.beta[1] - output3.beta[2]):int(output3.beta[1] + output3.beta[2])]), sum(Counts45Deg[int(output4.beta[1] - output4.beta[2]):int(output4.beta[1] + output4.beta[2])]), sum(Counts75Deg[int(output5.beta[1] - output5.beta[2]):int(output5.beta[1] + output5.beta[2])]) ]), ActiveMeasurementTime)
CountRatesUnc = np.divide(np.sqrt(np.array([ sum(Counts0Deg[int(output2.beta[1] - output2.beta[2]):int(output2.beta[1] + output2.beta[2])]), sum(Counts30Deg[int(output3.beta[1] - output3.beta[2]):int(output3.beta[1] + output3.beta[2])]), sum(Counts45Deg[int(output4.beta[1] - output4.beta[2]):int(output4.beta[1] + output4.beta[2])]), sum(Counts75Deg[int(output5.beta[1] - output5.beta[2]):int(output5.beta[1] + output5.beta[2])]) ])), ActiveMeasurementTime)


# Create a squared cosine fit for the angular dependency of the cosmic muons:
model6 = odr.Model(KosinusQuadrat)
data6 = odr.RealData(Angle, CountRates, sy=CountRatesUnc)
ODR6 = odr.ODR(data6, model6, beta0=[0.18])
output6 = ODR6.run()


#------------------------------------------------------------------------------
# Show results:
# Time calibration:
plt.figure("Time calibration", figsize=(10, 7.5))
plt.errorbar(PeakPositions, dt, xerr=PeakPositionsUnc, capsize=5., ecolor="C0", fmt=".C0", zorder=2, label="Data points")
plt.plot(np.linspace(0., 4000., 2), line(output0.beta, np.linspace(0., 4000., 2)), "-r", linewidth=3.5, zorder=3, label="Linear fit with"+"\n"+r"$m=(2.65463\pm 0.00011)\cdot 10^{-3}\,\mu$s,"+"\n"+r"$n=(-0.7299\pm 0.0002)\,\mu$s and"+"\n"+"$RV=1.8385980272423096\cdot 10^{-7}$")
plt.legend(loc="best", fontsize=18)
plt.xlabel(r"Channel $K$", fontsize=20)
plt.ylabel(r"Time interval $\Delta$t $\left[ \mu s\right]$", fontsize=20)
plt.xlim(0., 4000.)
plt.ylim(-1., 11.)
plt.yticks(np.linspace(-1., 11., 13, endpoint=True))
plt.tick_params(labelsize=16)
plt.grid(True)
plt.title("Time calibration", fontsize=20)
plt.savefig("TimeCalibration.pdf")
plt.show()
print("Steigung: m = (", output0.beta[0], "+/-", output0.sd_beta[0], ")")
print("y-Achsenabschnitt: n = (", output0.beta[1], "+/-", output0.sd_beta[1], ")")
print(output0.res_var)

# Mean lifetime and decay measurement:
plt.figure("Mean lifetime and decay measurement", figsize=(10, 7.5))
plt.axvline(x=Time[135], color="C1", linestyle="--", linewidth=2.5, zorder=4)
plt.axvline(x=Time[8008], color="C1", linestyle="--", linewidth=2.5, zorder=4)
plt.errorbar(Time, BinnedCounts, xerr=TimeUnc, yerr=BinnedCountsUnc, capsize=5., ecolor="C0", fmt=".C0", zorder=2, label="Binned data points")
plt.plot(Time[136:8008], exponential(output1.beta, Time)[136:8008], "-r", linewidth=3.5, zorder=3, label=r"Exponential fit with $\tau =(2.103\pm 0.007)\,\mu$s"+"\n"+r"and $RV=1.067631397018964$")
plt.legend(loc="best", fontsize=18)
plt.xlabel(r"Time $t$ $\left[ \mu s\right]$", fontsize=20)
plt.ylabel(r"Counts $N$", fontsize=20)
plt.xlim(-2., 22.)
plt.xticks(np.linspace(-2., 22., 13, endpoint=True))
plt.tick_params(labelsize=16)
plt.grid(True)
plt.savefig("MeanLifetime.pdf")
plt.show()
print(output1.beta)
print(output1.sd_beta)
print(output1.res_var)


#------------------------------------------------------------------------------
# Show results:
# Energy loss spectrum at 0 degrees:
plt.figure(r"Energy loss spectrum at $\theta =0\,$°", figsize=(10, 7.5))
plt.axvline(x=channel[399], color="C1", linestyle="--", linewidth=2.5, zorder=4)
plt.errorbar(channel, BinnedCounts0Deg/ActiveMeasurementTime[0], xerr=np.repeat(0.5/np.sqrt(3), n), yerr=BinnedCounts0DegUnc/ActiveMeasurementTime[0], capsize=5., ecolor="C0", fmt=".C0", zorder=2, label="Binned data points")
plt.plot(channel, LandauDistribution(output2.beta, channel), "-r", linewidth=3.5, zorder=3, label="Landau fit with"+"\n"+r"$K_0=481.61\pm 0.13$,"+"\n"+r"$w=41.56\pm 0.08$ and"+"\n"+"$RV=0.4335934292670435$")
plt.legend(loc="best", fontsize=18)
plt.xlabel(r"Channel $K$", fontsize=20)
plt.ylabel(r"In time normalized counts $N_T$ $\left[\frac{1}{s}\right]$", fontsize=20)
plt.xlim(0., 1500.)
plt.xticks(np.linspace(0., 1400., 8, endpoint=True))
plt.tick_params(labelsize=16)
plt.grid(True)
plt.savefig("EnergySpectrumAt0Degrees.pdf")
plt.show()
print(output2.beta)
print(output2.sd_beta)
print(output2.res_var)

# Energy loss spectrum at 30 degrees:
plt.figure(r"Energy loss spectrum at $\theta =30\,$°", figsize=(10, 7.5))
plt.axvline(x=channel[399], color="C1", linestyle="--", linewidth=2.5, zorder=4)
plt.errorbar(channel, BinnedCounts30Deg/ActiveMeasurementTime[1], xerr=np.repeat(0.5/np.sqrt(3), n), yerr=BinnedCounts30DegUnc/ActiveMeasurementTime[1], capsize=5., ecolor="C0", fmt=".C0", zorder=2, label="Binned data points")
plt.plot(channel, LandauDistribution(output3.beta, channel), "-r", linewidth=3.5, zorder=3, label="Landau fit with"+"\n"+r"$K_0=485.01\pm 0.15$,"+"\n"+r"$w=42.80\pm 0.08$ and"+"\n"+"$RV=0.4403359397479121$")
plt.legend(loc="best", fontsize=18)
plt.xlabel(r"Channel $K$", fontsize=20)
plt.ylabel(r"In time normalized counts $N_T$ $\left[\frac{1}{s}\right]$", fontsize=20)
plt.xlim(0., 1500.)
plt.xticks(np.linspace(0., 1400., 8, endpoint=True))
plt.tick_params(labelsize=16)
plt.grid(True)
plt.savefig("EnergySpectrumAt30Degrees.pdf")
plt.show()
print(output3.beta)
print(output3.sd_beta)
print(output3.res_var)

# Energy loss spectrum at 45 degrees:
plt.figure(r"Energy loss spectrum at $\theta =45\,$°", figsize=(10, 7.5))
plt.axvline(x=channel[399], color="C1", linestyle="--", linewidth=2.5, zorder=4)
plt.errorbar(channel, BinnedCounts45Deg/ActiveMeasurementTime[2], xerr=np.repeat(0.5/np.sqrt(3), n), yerr=BinnedCounts45DegUnc/ActiveMeasurementTime[2], capsize=5., ecolor="C0", fmt=".C0", zorder=2, label="Binned data points")
plt.plot(channel, LandauDistribution(output4.beta, channel), "-r", linewidth=3.5, zorder=3, label="Landau fit with"+"\n"+r"$K_0=485.98\pm 0.17$,"+"\n"+r"$w=44.06\pm 0.10$ and"+"\n"+"$RV=0.40113826222667925$")
plt.legend(loc="best", fontsize=18)
plt.xlabel(r"Channel $K$", fontsize=20)
plt.ylabel(r"In time normalized counts $N_T$ $\left[\frac{1}{s}\right]$", fontsize=20)
plt.xlim(0., 1500.)
plt.xticks(np.linspace(0., 1400., 8, endpoint=True))
plt.tick_params(labelsize=16)
plt.grid(True)
plt.savefig("EnergySpectrumAt45Degrees.pdf")
plt.show()
print(output4.beta)
print(output4.sd_beta)
print(output4.res_var)

# Energy loss spectrum at 75 degrees:
plt.figure(r"Energy loss spectrum at $\theta =75\,$°", figsize=(10, 7.5))
plt.axvline(x=channel[449], color="C1", linestyle="--", linewidth=2.5, zorder=4)
plt.errorbar(channel, BinnedCounts75Deg/ActiveMeasurementTime[3], xerr=np.repeat(0.5/np.sqrt(3), n), yerr=BinnedCounts75DegUnc/ActiveMeasurementTime[3], capsize=5., ecolor="C0", fmt=".C0", zorder=2, label="Binned data points")
plt.plot(channel, LandauDistribution(output5.beta, channel), "-r", linewidth=3.5, zorder=3, label="Landau fit with"+"\n"+r"$K_0=491.1\pm 1.4$,"+"\n"+r"$w=49.9\pm 0.6$ and"+"\n"+"$RV=1.9681241099059543$")
plt.legend(loc="best", fontsize=18)
plt.xlabel(r"Channel $K$", fontsize=20)
plt.ylabel(r"In time normalized counts $N_T$ $\left[\frac{1}{s}\right]$", fontsize=20)
plt.xlim(0., 1500.)
plt.xticks(np.linspace(0., 1400., 8, endpoint=True))
plt.tick_params(labelsize=16)
plt.grid(True)
plt.savefig("EnergySpectrumAt75Degrees.pdf")
plt.show()
print(output5.beta)
print(output5.sd_beta)
print(output5.res_var)

# Angular dependency of the cosmic muons:
plt.figure("Angular dependency of the cosmic muons", figsize=(10, 7.5))
plt.plot(Angle, CountRates, ".k")
plt.errorbar(Angle, CountRates, yerr=CountRatesUnc, capsize=5., ecolor="C0", fmt=".C0", zorder=2, label="Data points")
plt.plot(np.linspace(0., 90., 1000), KosinusQuadrat(output6.beta, np.linspace(0., 90., 1000)), "-r", linewidth=3.5, zorder=3, label="Squared cosine fit with"+"\n"+r"$A=(0.179\pm 0.006)\,\frac{1}{s}$ and"+"\n"+"$RV=19.896046077201788$")
plt.legend(loc="best", fontsize=18)
plt.xlabel(r"Zenith angle $\theta$ $\left[ °\right]$", fontsize=20)
plt.ylabel(r"Count rate $R$ $\left[\frac{1}{s}\right]$", fontsize=20)
plt.xlim(0., 90.)
plt.ylim(0., 0.2)
plt.xticks(np.linspace(0., 90., 10, endpoint=True))
plt.yticks(np.linspace(0., 0.2, 9, endpoint=True))
plt.tick_params(labelsize=16)
plt.grid(True)
plt.savefig("AngularDependency.pdf")
plt.show()
print(output6.beta)
print(output6.sd_beta)
print(output6.res_var)