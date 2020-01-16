import uncertainties as unc
import math

# definitions without error
lightspeed = c0 = c = 299792458 # m/s
boltzmann_constant = k_B = 1.380649e-23  # J K-1 [0]
planck_constant = h = 6.62607015e-34 # J s
avogadro_constant = N_A = 6.02214076e23 # mol-1
electron_charge = elementary_charge = e = 1.602176634e-19 # C
rad = 360 / 2 / math.pi
grad = 1/rad
kelvin = 273.15 # 0 degree = kelvin

# measurements with error
vacuum_permeability = mu0 = 1.25663706212e-6# 
electron_mass = m_e = 9.1093837015e-31 # kg
atomic_mass_unit = dalton = u = 1.66053906660e-27 # kg
barn = 1E-28 # m2
g_erde = 9.81 # m/s^2

# derived constants
vacuum_permitivity = eps0 = 1 / (mu0 * c**2)
reduced_planck_constant = hbar = h/(2*math.pi)
coulomb_constant = k_e = 1 / (4*math.pi*eps0)
finestructure_constant = alpha = k_e * e**2 / (hbar * c0)
rydberg_constant = R_infty = m_e * e**4 / (8*eps0**2*h**3*c)
bohr_radius = a_0 = hbar / (alpha * m_e*c)
electron_radius = r_e = alpha**2 * a_0

# units
electronvolt = eV = e


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
