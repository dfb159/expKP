import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as odr
from scipy import constants


# Read data:
counts = np.loadtxt("messungA1.txt", dtype = int, delimiter = " ", usecols = 1)
fakeVelocity = np.loadtxt("velocity1_smoothed.txt", dtype = int, delimiter = " ", usecols = 1)


# Initialize global variables:
n = 256
halfChannel = np.linspace(0, n-1, num=n)


# Some functions to fit and calculate:
def absKosinus(parameter, x):
    A, phi = parameter
    return A * np.absolute(np.cos((np.pi / 256.) * (x + phi)))

def minusKosinus(A, phi, x):
    return -A * np.cos((np.pi / 256.) * (x + phi))

def Sinus(A, phi, x):
    return A * np.sin((np.pi / 256.) * (x + phi))

def minusKosinusUnc(A, phi, A_err, phi_err, x):
    return np.sqrt((minusKosinus(A_err, phi, x))**2 + (Sinus((A*np.pi*phi_err)/256., phi, x))**2)

def Gaussian(parameter, x):
    A, mu, sigma, B = parameter
    return A * np.exp(-.5 * ((x - mu)/sigma)**2) + B

def line(parameter, x):
    m, n = parameter
    return m * x + n

def calculateFWHM(sigma):
    return 2 * np.sqrt(2 * np.log(2)) * sigma

def calculateMesseffekt(A, B):
    return -A / B

def calculateMesseffektUnc(A, B, A_err, B_err):
    return calculateMesseffekt(A, B) * np.sqrt((A_err / A)**2 + (B_err / B)**2)

def f_A_StainlessSteel(T):
    k_B = 8.617342e-05
    Theta_D = 450.
    p = 14.41295e03
    M = 56.935398 * 931.494013e6
    return np.exp(((-3. * p**2) / (4. * M * k_B * Theta_D)) * (1 + ((2. * (np.pi * T)**2) / (3. * Theta_D**2))))

def f_A_StainlessSteel_Unc(T, T_err):
    k_B = 8.617342e-05
    Theta_D = 450.
    p = 14.41295e03
    M = 56.935398 * 931.494013e6
    return (T * T_err * f_A_StainlessSteel(T) * (np.pi * p)**2)/(k_B * M * Theta_D**3)

def Gamma_korr(T):
    Gamma_nat = 0.097
    sigma0 = 2.38e-18
    eta = 0.0219
    n = (0.0079 * (constants.value("speed of light in vacuum"))**2)/(56.845 * 931.494013e06 * constants.value("elementary charge"))
    d = 0.0025
    return (2.02 + 0.29*sigma0*eta*n*d*f_A_StainlessSteel(T) - 0.005*(sigma0*eta*n*d*f_A_StainlessSteel(T))**2)*Gamma_nat

def Gamma_korr_Unc(T, T_err):
    Gamma_nat = 0.097
    sigma0 = 2.38e-18
    eta = 0.0219
    n = (0.0079 * (constants.value("speed of light in vacuum"))**2)/(56.845 * 931.494013e06 * constants.value("elementary charge"))
    d = 0.0025
    return (0.29*sigma0*eta*n*d - 0.01*f_A_StainlessSteel(T)*(sigma0*eta*n*d)**2)*Gamma_nat*T_err

def calculateDeltaRR(v_iso):
    Z = 26.
    S = 1.33
    R = 1.3e-15 * 57**(1./3.)
    E_gamma = 14.41295e03 * constants.value("elementary charge")
    a0 = 0.0529e-09
    Phi0_ss_squard = 11882.4/(a0**3)
    Phi0_Pd_squard = 11881.8/(a0**3)
    return (5.*E_gamma*constants.value("electric constant")*np.abs(v_iso))/(S*Z*constants.value("speed of light in vacuum")*(Phi0_ss_squard - Phi0_Pd_squard)*(R*constants.value("elementary charge"))**2)

def calculateSingleTemperatureValueUnc(m_err, n_err, x_err, m, x):
    return np.sqrt((x * m_err)**2 + (n_err)**2 + (m * x_err)**2)


# Formulas for the velocity and its uncertainty with an adjustment of the scale:
def calculateVelocity(N_k, frequency, wavelength, RunsPerMeasurement, N):
    return ((N * N_k * frequency * wavelength)/(2. * RunsPerMeasurement)) * 1e-06

def calculateVelocityUnc(N_k, frequency, wavelength, RunsPerMeasurement, wavelength_err, N):
    return calculateVelocity(N_k, frequency, wavelength, RunsPerMeasurement, N) * np.sqrt((1/N) + (wavelength_err/wavelength)**2)


# Kills the reflection:
# For messungA1:
combinedCounts = np.zeros(n)
for i in range(0, n):
    combinedCounts[i] = counts[n - 1 - i] + counts[n + i]
# For velocity1_smoothed:
combinedFakeVelocity = np.zeros(n)
for i in range(0, n):
    combinedFakeVelocity[i] = fakeVelocity[n - 1 - i] + fakeVelocity[n + i]


# Calculate the acual velocity and its uncertainty:
velocity = calculateVelocity(n, 24., 635., 20000., combinedFakeVelocity)
velocityUnc = calculateVelocityUnc(n, 24., 635., 20000., 5., combinedFakeVelocity)


# Fitting velocity spectrum data:
model0 = odr.Model(absKosinus)
data0 = odr.RealData(halfChannel, velocity, sy=velocityUnc)
ODR0 = odr.ODR(data0, model0, beta0=[2.06, 0.5])
output0 = ODR0.run()


# Change the x axis from channel to real velocity:
realVelocity = minusKosinus(output0.beta[0], output0.beta[1], halfChannel)


# The uncertainty of a real velocity value is the horizontal uncertainty of the corresponding data point:
realVelocityUnc = minusKosinusUnc(output0.beta[0], output0.beta[1], output0.sd_beta[0], output0.sd_beta[1], halfChannel)


# Fitting the spectrum of Messung A:
model = odr.Model(Gaussian)
data = odr.RealData(realVelocity, combinedCounts, sx=realVelocityUnc, sy=np.sqrt(combinedCounts))
ODR = odr.ODR(data, model, beta0=[-80., -0.25, 0.25, 200.])
output = ODR.run()


# Temperature fit and calculation of the theoretical FWHM:
hours = []
minutes = []
minutesUnc = np.repeat(, 18)
time = np.zeros(18)
temperature = [19.8, 20., 20., 20.2, 20.6, 20.8, 21.2, 21.4, 21.4, 21.6, 21.6, 21.8, 21.8, 22., 22., 22.2, 22.2, 22.2]
temperatureUnc = np.repeat(0.1/np.sqrt(3), 18)
for i in range(0, 18):
    time[i] = minutes[i] / 60. + hours[i]
    temperature[i] = 273.15 + temperature[i]
model2 = odr.Model(line)
data2 = odr.RealData(time, temperature, sx=minutesUnc, sy=temperatureUnc)
ODR2 = odr.ODR(data2, model2, beta0=[, ])
output2 = ODR2.run()
output2.pprint()


# Show results:
# Sixth plot:
plt.figure("|Velocity| Spectrum")
plt.errorbar(halfChannel, velocity, yerr=velocityUnc, fmt="none", label="Messwerte")
plt.plot(halfChannel, absKosinus(output0.beta, halfChannel), "-r", label="Kalibrierungsfit")
plt.legend(loc="best")
plt.title("|Velocity| Spectrum", fontsize=18)
plt.xlabel("Channel", fontsize=16)
plt.ylabel("|Velocity| (in mm/s)", fontsize=16)
plt.grid(True)
plt.show()

print("A =", output0.beta[0], "+/-", output0.sd_beta[0])
print("phi =", output0.beta[1], "+/-", output0.sd_beta[1])

# Eighth plot:
plt.figure("Spectrum Messung A")
plt.errorbar(realVelocity, combinedCounts, xerr=realVelocityUnc, yerr=np.sqrt(combinedCounts), fmt="none", label="Messwerte")
plt.plot(realVelocity, Gaussian(output.beta, realVelocity), "-r", label="Gaußsche Anpassungskurve")
plt.legend(loc="best")
plt.title("Spectrum Messung A", fontsize=18)
plt.xlabel("Velocity (in mm/s)", fontsize=16)
plt.ylabel("Counts", fontsize=16)
plt.grid(True)
plt.show()


# Show the fit parameters and the values, that need to be calculated:
print(output.beta)
print(output.sd_beta)
print(np.sqrt(np.diag(output.cov_beta * output.res_var)))
print("Messeffekt: -A/B =", calculateMesseffekt(output.beta[0], output.beta[3]), "+/-", calculateMesseffektUnc(output.beta[0], output.beta[3], output.sd_beta[0], output.sd_beta[3]))
print("Halbwertsbreite: FWHM =", calculateFWHM(output.beta[2]), "+/-", calculateFWHM(output.sd_beta[2]))
print("Isomerieverschiebung: v_iso =", output.beta[1], "+/-", output.sd_beta[1])
print("Relative Aenderung des Kernladungsradius: dR/R =", calculateDeltaRR(output.beta[1]), "+/-", calculateDeltaRR(output.sd_beta[1]))