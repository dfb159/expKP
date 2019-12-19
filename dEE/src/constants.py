import uncertainties as unc
import math

c0 = 299792458 # m/s
k_B = unc.ufloat_fromstr("1.38064852(79)e-23") # J K-1 [0]
hbar = unc.ufloat_fromstr("4.135667662(25)e-15") / 2 / math.pi # eV s [0]
barn = 1E-28 # m2
kelvin = 273.15 # kelvin
g_erde = 9.81 # m/s^2
rad = 360 / 2 / math.pi
grad = 1/rad
terra = 1e12
giga = 1e9
mega = 1e6
kilo = 1e3
dezi = 0.1
centi = 0.01
milli = 1e-3
micro = 1e-6
nano = 1e-9
pico = 1e-12
femto = 1e-15
