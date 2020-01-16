 # %% Importanweisungen
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors as mcolors
import math
import numpy as np
import statistics as stat
import scipy as sci
import uncertainties as unc
import uncertainties.unumpy as unp
import uncertainties.umath as umath
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as unv
from uncertainties.unumpy import std_devs as usd
import pandas as pd
from collections.abc import Iterable
import seaborn

#importlib.import_module("constants")
import constants as C
import latex
import fitting as fit
import plotting as plot

# %% Konstanten fuer einheitliche Darstellung

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

root = "../"
data_out = root + "dat/"
raw_in = root + "raw/"

# %% import der Messwerte

def stats(n):
    if isinstance(n, Iterable):
        return unp.uarray(n, np.sqrt(abs(np.array(n))+1))
    else:
        return unc.ufloat(n, np.sqrt(abs(n)+1))
    
folien_kalib = []
for m, w in zip(["kalibrierung_E", "folie1", "folie2", "folie3"], [5485 / 5056,1,1,1]):
    E, C1, C2 = np.loadtxt(raw_in + "versuch2/%s/histograms_energy.dat" % m, unpack=True)
    E = unp.uarray(E, (E[1]-E[0]) / np.sqrt(12)) * w
    C1 = stats(C1) / max(C1)
    folien_kalib.append((E, C1))
    
E, dE1, dE2, dE = np.loadtxt(root + "res/dEE-mylar.csv", unpack=True, skiprows=1, delimiter=",")
mylar_table = E, dE

folien_dat = []
xs = np.zeros((40000))
E_min = []
E_max = []
dE_min = []
dE_max = []
names = list(map(lambda f: "messung_folie" + f, ["1_0grad", "1_15grad", "1_30grad", "1_45grad", "1_60grad", "2_0grad", "2_15grad", "2_30grad", "2_45grad", "3_0grad", "3_15grad"]))
for m in names:
    f = []
    data = np.loadtxt(raw_in + "versuch2/%s/histograms_energy2d.dat" % m)
    for E, dE, n in data:
        if n > 0:
            f.append((E+dE,dE,n))
    folien_dat.append(list(map(np.array, list(zip(*f)))))
    E, dE, n = np.array(data).T
    E_min.append(min(E))
    E_max.append(max(E))
    dE_min.append(min(dE))
    dE_max.append(max(dE))
    xs += n

E_min = min(E_min)
E_max = max(E_max)
dE_min = min(dE_min)
dE_max = max(dE_max)
xs = xs.reshape((200,200))

# %% 3. Dickenbestimmung

def gauss(x, A, x0, s, y0):
    exp = unp.exp if any(map(lambda x: usd(x) != 0, [A, x0, s, y0])) or any(usd(x) != 0) else np.exp
    return A * exp(-(x-x0)**2/s**2/2) + y0

def gauss2(x, A1, x1, s1, A2, x2, s2, y0):
    return gauss(x, A1, x1, s1, 0) + gauss(x, A2, x2, s2, 0) + y0

fig = plt.Figure(figsize=plot.fullscreen)

x0 = [5480, 4508, 3370, 2021]
r = [(4700,5800),(3500, 4850),(2300,3800),(1000,2500)]
label=["ohne", "1.", "2.", "3."]
ps = []
for i, ((x, y), (r1, r2)) in enumerate(zip(folien_kalib,r)):
    width = unv(x[1] - x[0]) * 0.8
    xfit, yfit, p = fit.fitspaceXY(x,y, gauss2,  [200, x0[i], 100, 300, x0[i]-350, 135, 0], num=500)
    ps.append(p)
    plot.fit(xfit, yfit, color="C%i"%i, linestyle="None", sigma=3)
    plot.error(x, y, label="%s Folie" % label[i], color="C%i"%i, capsize=3, linewidth=1, errorevery=1, fmt=".", left=r1, right=r2)
#    plt.bar(unv(x), unv(y), label="%s Folie" % label[i], color="C%i"%i, width=width, zorder=25, alpha=0.9)

plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    right=False,      # ticks along the bottom edge are off
    left=False,         # ticks along the top edge are off
    labelleft=False # labels along the bottom edge are off
    )
plot.params(xlabel="Energie $E$ [$keV$]", ylabel="normierte Zählrate", grid=True)#,xlim=(4500, 5000), ylim=(0,200))
plot.save(data_out+"m3_foliendicke")
plot.finish()

latex.table(np.array(ps).T, data_out, "m3_fitdaten", header=["","ohne", "1. Folie", "2. Folie", "3. Folie"], leader=["$A_1$", "$E_1$", "$\\sigma_1$", "$A_2$", "$E_2$", "$\\sigma_2$", "$A_0$"], units=["", "\\kilo\\electronvolt", "\\kilo\\electronvolt", "", "\\kilo\\electronvolt", "\\kilo\\electronvolt", ""], horizontal=True)

p = np.array([ps[i][1] + unc.ufloat(0, np.sqrt(unv(ps[i][2])**2 + usd(xs[i,2])**2)) for i in range(len(ps))])
print(p)

# %% Interpolation of mylar dataset
mylar = sci.interpolate.interp1d(*mylar_table, kind="cubic")
dx=1e-4
E0 = 5485.56
E, x = E0, 0
dat = [(E,x)]
while 1000 <= E:
    E -= mylar(E) * dx
    x += dx
    dat.append((E,x))

x, y = list(zip(*dat))
findD = np.vectorize(sci.interpolate.interp1d(x,y, kind="cubic"))
findE = np.vectorize(sci.interpolate.interp1d(y,x, kind="cubic"))

#%%

p2 = p[1:]
dicke = findD(unv(p2))
unc1 = findD(unv(p2) + usd(p2))
unc2 = findD(unv(p2) - usd(p2))
dicke = unp.uarray(dicke, (abs(dicke - unc1) + abs(dicke - unc2))/2)
for i,(v,e) in enumerate(zip(dicke,p2)):
    latex.SI(v, "\\micro\\meter", data_out, "folie_dicke_%i" % (i+1), show=True)
    latex.SI(e.format(".0f"), "\\kilo\\electronvolt", data_out, "folie_energie_%i" % (i+1), bonus="separate-uncertainty=true", show=True)

#%%

def BBFormel(E_kin, z, M, Z, A, I, X0, X1, C0, a, m, rho):
    if E_kin <= 0:
        return 0
    gamma = 1 + E_kin / (M*C.c**2)
    beta = np.sqrt(1-1/gamma**2)
    K = 2*math.pi * C.N_A * C.r_e**2 * C.m_e * C.c**2 # J m2 kg-1
    v = beta * C.c
    W_max = 2 * C.m_e * v**2 * gamma**2 # for M >> m_e
    eta = beta * gamma
    X = np.log10(eta)
    delta = 0 if X <= X0 else 4.6052 * X + C0 + a*(X1 - X)**m
    CC = (0.422377 / eta**2 + 0.0304043 / eta**4 - 0.00038106 / eta**6) * 1e-6 * I**2 + (3.850190 / eta**2 - 0.1667989 / eta**4 + 0.00157955 / eta**6) * 1e-9 * I**3
    logterm = np.log(2*C.m_e * gamma**2*v**2*W_max/I**2)
    parenthesis = logterm - 2*beta**2 - delta - 2*CC/Z
    constant = -K * Z * z**2 / (A*beta**2)
    erg = constant * rho * parenthesis # J m-1
    return erg if erg > 0 else 0

def silizium(E_kin, z, M):
    Z = 14
    A = 28
    rho = 2.3212 * C.milli / C.centi**3 # kg m-3
    I = 173 # eV
    C0 = -4.44
    a = 0.1492
    m = 3.25
    X0 = 0.2014
    X1 = 2.87
    return BBFormel(E_kin, z, M, Z, A, I, X0, X1, C0, a, m, rho) # J m-1

def BetheBloch(E_kin, M, z): # kinetische Energie und Ruhemasse in MeV, z: Ladung in e
    if E_kin <= 0:
        return 0
    K = 0.1535 #MeV cm^2 / g
    rho = 2.329#36  #g / cm^3 , bei 20Â°C, aus N. N. Greenwood, A. Earnshaw: Chemie der Elemente. 1988, ISBN 3-527-26169-9, S. 426.
    Z = 14
    A = 28.085
    I = (9.76 + 58.8 * Z**-1.19) * Z #eV
    C_0 = -4.44
    a = 0.1492
    m = 3.25
    X_1 = 2.87
    X_0 = 0.2014
    m_e = 0.51099895 #MeV, aus CODATA Recommended Values. National Institute of Standards and Technology, abgerufen am 20. Mai 2019.
    
    gamma = 1 + E_kin / M
    beta = np.sqrt(1 - 1 / gamma ** 2)
    eta = gamma * beta
    X = np.log10(eta)
    s = m_e / M
    W_max = 2*m_e * eta**2
    W_max2 = W_max / (1 + 2*s*np.sqrt(1+eta**2)+s**2)
                 
    delta = 0 if X < X_0 else 4.6052 * X + C_0 + a * (X_1 - X)**m if X_0 <= X <= X_1 else 4.6052 * X + C_0
    CC = 0 if eta < 0.1 else (0.422377 * eta ** (-2) + 0.0304043 * eta ** (-4) - 0.00038106 * eta ** (-6)) * 10 ** (-6) * I ** 2 + (3.850190 * eta ** (-2) - 0.1667989 * eta ** (-4) + 0.00157955 * eta ** (-6)) * 10 ** (-9) * I ** 3 # I in eV
        
    erg = K * rho * Z / A * z**2 / beta ** 2 * (np.log(W_max*W_max2 / I**2 * 10 ** 12) - 2 * beta ** 2 - delta - 2 * CC / Z) # MeV cm-1
    return erg if erg > 0 else 0
    
d = 8.5257e-4 # cm
def E_loss(E_kin, M, z, num=500):
    dx = d / num
    E = E_kin
    for i in range(num):
        E -= BetheBloch(E, M, z) * dx
        if E < 0:
            return E_kin
    return E_kin - E

def E_interpolate(M, z, num=500):
    x = np.linspace(0, 20, num) # MeV
    y = np.vectorize(lambda E: E_loss(E, M, z))(x) # MeV
    f = sci.interpolate.interp1d(x*1000,y*1000, kind="cubic") # keV
    return np.vectorize(lambda E: gainUnc(E,f)) # keV

def gainUnc(X, f):
    if usd(X) == 0:
        return f(X)
    X1, X2 = unv(X), usd(X)
    f1 = f(X1)
    f2 = f(X1+X2)
    f3 = f(X1-X2)
    return unc.ufloat(f1, (abs(f1 - f2) + abs(f1 - f3))/2)
    

#proton = np.vectorize(lambda E: silizium(E, 1, 1.67262192369e-27))
#deuteron = np.vectorize(lambda E: silizium(E, 1, 3.3435837724e-27))
#triton = np.vectorize(lambda E: silizium(E, 1, 5.0073567446e-27))
#helium3 = np.vectorize(lambda E: silizium(E, 2, 3.0160293 * C.u))
#alpha = np.vectorize(lambda E: silizium(E, 2, 6.6446573357e-27))
#lithium = np.vectorize(lambda E: silizium(E, 3 , 6.94 * C.u))
    
betheblochformeln = [
        E_interpolate(938.272, 1),
        E_interpolate(1875.61, 1),
        E_interpolate(2809.432, 1),
        E_interpolate(2809.414, 2),
        E_interpolate(3727.379, 2),
        E_interpolate(6465.5, 3),]

#%%

plot.setup()

#for E,dE,n in folien_dat:
#    plt.scatter(E,dE,s=5*n, alpha=0.3)

data = []
for k in folien_dat:
    data.extend(tuple(zip(*k)))
data = np.array(data)

N, M = 100, 50
E_max2 = max(data[:,0])
E_min2 = min(data[:,0])
dE_max2 = max(data[:,1])
dE_min2 = min(data[:,1])
array = np.zeros((N, M), int)
# convert to array
for E, dE, n in data:
    i = int(N * (E - E_min2) / (E_max2-E_min2))
    j = int(M * (dE - dE_min2) / (dE_max2 - dE_min2))
    i = max(min(i,N-1), 0)
    j = max(min(j,M-1), 0)
    array[i,j] += 1

#add all channels
    
data = np.zeros((200,400))
data[:,:200] = xs.T[::-1,:]
for i in range(200):
    data[i] = np.roll(data[i],200-i)

#seaborn.heatmap(xs.T, mask=(xs.T==0), cmap="plasma", vmax=100)#, extent=(E_min,E_max, dE_min, dE_max))
my_cmap = matplotlib.cm.get_cmap('plasma')
my_cmap.set_under((0,0,0,0))
data = xs.T[::-1,:]
X, Y = np.meshgrid([E_min + 1.3*i * (E_max-E_min) / 200 for i in range(200)], [dE_max - i * (dE_max-dE_min) / 200 for i in range(200)])
plt.imshow(data, cmap=my_cmap, vmax=50, vmin=0.1, extent=(E_min,E_max*1.3, dE_min, dE_max), zorder=15,interpolation="bicubic")
#plt.pcolormesh(X, Y, data, cmap=my_cmap, vmax=50, vmin=0.1, zorder=15, shading="flat")

x = np.linspace(0, 8000, 100) # keV
name = ["Proton", "Deuteron", "Triton", "Helium-3", "Alpha", "Lithium"]
for g, m in zip(betheblochformeln,name):
    ydraw = g(x) # MeV cm-1
    plt.plot(x, ydraw, label=m,zorder = 25)

plot.params(xlabel=u"kinetische Energie $E_{kin}$ [keV]", ylabel=u"Energieverlust $dE$ [keV]", xlim=(0,7500), ylim=(0,3000), grid=True)
plot.params(xlabel=u"kinetische Energie $E_{kin}$ [keV]", ylabel=u"Energieverlust $dE$ [keV]", xlim=(0,10000), ylim=(0,10000), grid=True)
plot.save(data_out + "energieverlust")

#%%

def debeta(E, dE, n, M): # E and M in MeV
    gamma = 1 + E / M
    beta = np.sqrt(1-1/gamma**2)
    return (dE * beta**2, n)

debetaformeln = [
        lambda E, dE, n: debeta(E, dE, n, 938.272),
        lambda E, dE, n: debeta(E, dE, n, 1875.61),
        lambda E, dE, n: debeta(E, dE, n, 2809.432),
        lambda E, dE, n: debeta(E, dE, n, 2809.414),
        lambda E, dE, n: debeta(E, dE, n, 3727.379),
        lambda E, dE, n: debeta(E, dE, n, 6465.5),]

E0 = 5485.56

# sort through data
N = 20
m, n = 500, 1750
erg = []
bins = np.zeros((N), int)
for f in debetaformeln:
    d = [f(E,dE,n) for E, dE, n in data]
    erg.append(d)
    bins = np.zeros((N), int)
    for deb, k in d:
        i = int(N * (deb-m) / (n-m))
        if 0 <=i < N: # in range
            bins[i] += k
    x = np.linspace(m,n,N)
    w = (n-m)/N*0.8
    plt.bar(x,bins, width=w)
    plt.show()
        

# %%
    
x = np.linspace(0,100, 50)
y = np.exp(-(x-50)**2/10**2)
plt.bar(x,y)
