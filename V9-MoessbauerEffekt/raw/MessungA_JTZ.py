import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt

# Read data:
messungA = "messungA1.txt"
speedchannel = np.loadtxt(messungA, dtype = int, delimiter = " ", usecols = 0)
counts = np.loadtxt(messungA, dtype = int, delimiter = " ", usecols = 1)
velocity0 = "velocity0_ohne_probe_smoothed.txt"
channel = np.loadtxt(velocity0, dtype = int, delimiter = " ", usecols = 0)
velocity = np.loadtxt(velocity0, dtype = int, delimiter = " ", usecols = 1)

# Kills the reflection:
n1 = 511
n2 = 256
combinedcounts = np.zeros(n2)
halfspeedchannel = np.zeros(n2)
for i in range(0, n2):
    combinedcounts[i] = counts[i] + counts[n1 - i]
    halfspeedchannel[i] = i

# First plot:
plt.figure("RawSpectrum")
plt.plot(speedchannel, counts, "ok")
plt.title("RawSpectrum", fontsize=18)
plt.xlabel("Channel", fontsize=14)
plt.ylabel("Counts", fontsize=14)
plt.grid(True)
plt.show()

# Second plot:
plt.figure("CombinedSpectrum")
plt.plot(halfspeedchannel, combinedcounts, "ok")
plt.title("CombinedSpectrum", fontsize=18)
plt.xlabel("Channel", fontsize=14)
plt.ylabel("Counts", fontsize=14)
plt.grid(True)
plt.show()

# Third plot:
plt.figure("FilteredSpeedSpectrum")
plt.plot(channel, velocity, "ob")
plt.title("FilteredSpeedSpectrum", fontsize=18)
plt.xlabel("Channel", fontsize=14)
plt.ylabel("Speed", fontsize=14)
plt.grid(True)
plt.show()