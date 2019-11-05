import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Read data:
messungA = "messungA1.txt"
speedchannel = np.loadtxt(messungA, dtype = int, delimiter = " ", usecols = 0)
counts = np.loadtxt(messungA, dtype = int, delimiter = " ", usecols = 1)
velocity0 = "velocity1_smoothed.txt"
channel = np.loadtxt(velocity0, dtype = int, delimiter = " ", usecols = 0)
fake_velocity = np.loadtxt(velocity0, dtype = int, delimiter = " ", usecols = 1)


# Initialize global variables:
n = 256
halfspeedchannel = np.zeros(n)
for i in range(0, n):
    halfspeedchannel[i] = i


# Fit functions:
def absKosinus(x, A, phi):
    return A * np.absolute(np.cos((np.pi / 256.) * (x + phi)))

def gaussian(x, A, mu, sigma, B):
    return A * np.exp(-.5 * ((x - mu)/sigma)**2) + B


# Kills the reflection:
# For messungA1:
combinedcounts = np.zeros(n)
for i in range(0, n):
    combinedcounts[i] = counts[(n - 1) - i] + counts[n + i]
# For velocity1_smoothed:
combinedfake_velocity = np.zeros(n)
for i in range(0, n):
    combinedfake_velocity[i] = fake_velocity[(n - 1) - i] + fake_velocity[n + i]


# Calculate the acual velocity and its uncertainty:
def calculateVelocity(N, N_k, frequency, wavelength, RunsPerMeasurement):
    return (N * N_k * frequency * wavelength)/(2. * RunsPerMeasurement)

def calculateVelocityUnc(N, N_k, frequency, wavelength, RunsPerMeasurement, errorwavelength):
    return calculateVelocity(N, N_k, frequency, wavelength, RunsPerMeasurement) * np.sqrt((1/N) + (errorwavelength/wavelength)**2)

velocity = np.zeros(n)
velocityUnc = np.zeros(n)
for i in range(0, n):
    velocity[i] = calculateVelocity(combinedfake_velocity[i], n, 24., 635., 20000.)
    velocityUnc[i] = calculateVelocityUnc(combinedfake_velocity[i], n, 24., 635., 20000., 5.)


# Fitting speed spectrum data:
popt, pcov = curve_fit(absKosinus, halfspeedchannel, velocity, sigma=velocityUnc)
print("A = ", popt[0], "+/-", np.sqrt(pcov[0][0]))
print("phi = ", popt[1], "+/-", np.sqrt(pcov[1][1]))


# Change x axis from channel to speed:
def Kosinus(x, A, phi):
    return A * np.cos((np.pi / 256.) * (x + phi))

speed = np.zeros(n)
for i in range(0, n):
    speed[i] = Kosinus(halfspeedchannel[i], popt[0], popt[1])


# Fitting the spectrum of Messung A:



# Sixth plot:
plt.figure("Speed Spectrum")
plt.errorbar(halfspeedchannel, velocity, yerr=velocityUnc, fmt="none")
plt.plot(halfspeedchannel, absKosinus(halfspeedchannel, popt[0], popt[1]), "-r")
plt.title("Speed Spectrum", fontsize=18)
plt.xlabel("Channel", fontsize=14)
plt.ylabel("Speed", fontsize=14)
plt.grid(True)
plt.show()

# Second plot:
plt.figure("Combined Spectrum")
plt.plot(halfspeedchannel, combinedcounts, ".k")
plt.title("Combined Spectrum", fontsize=18)
plt.xlabel("Channel", fontsize=14)
plt.ylabel("Counts", fontsize=14)
plt.grid(True)
plt.show()

# Seventh plot:
plt.figure("Spectrum Messung A")
plt.plot(speed, combinedcounts, ".k")
plt.title("Spectrum Messung A", fontsize=18)
plt.xlabel("Speed (in nm/s)", fontsize=14)
plt.ylabel("Counts", fontsize=14)
plt.grid(True)
plt.show()