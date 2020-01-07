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

#importlib.import_module("constants")
import constants as C
import latex
import fitting as fit

# %% Konstanten fuer einheitliche Darstellung
sidescreen = (8, 6)
fullscreen = (10,6)
widescreen = (16,6)
flatscreen = (10,4)
fig_size = fullscreen
fig_legendsize = 14
fig_labelsize = 12
matplotlib.rcParams.update({'font.size': fig_labelsize})

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

root = "../"
data_out = root + "dat/"
raw_in = root + "raw/"

# %% import der Messwerte

def stat(n):
    if isinstance(n, Iterable):
        return unp.uarray(n, np.sqrt(abs(n)+1))
    else:
        return unc.ufloat(n, np.sqrt(abs(n)+1))

messung1 = pd.read_csv(raw_in + "messung1/messung2.csv")
dt = unp.uarray(messung1["dt"], 0.025 / np.sqrt(3))
counts = stat(messung1["#"])
messung1 = dt, counts

messung3 = pd.read_csv(raw_in + "messung3/messung3.csv")
winkel = unp.uarray(messung3["Winkel"], 2/np.sqrt(6))
time = np.array(messung3["Messzeit"])
counts = stat(messung3["Counts"]) / time
messung3 = winkel, counts

# %% 1. Verzögerungsdauer

def gauss(x, A, x0, sigma, y0):
    return A * np.exp(-((x-x0)/sigma)**2/2) + y0

def gaussUnc(x,A,x0,sigma,y0):
    return A * unp.exp(-((x-x0)/sigma)**2/2) + y0

fig = plt.Figure(figsize=fullscreen)

dt, counts = messung1
plt.errorbar(unv(dt), unv(counts), xerr=usd(dt), yerr=usd(counts), fmt=" ", color="C0", zorder=10, label="Messung")

p0 = [1200, 0.333, 0.1, 0]
wechselT = 0.199
xdata, ydata = zip(*[(t,c) for t,c in zip(dt,counts) if t >= wechselT])
p = (A,x0,d,y0) = fit.fitXY(xdata, ydata, gauss, p0)
eff = gaussUnc(0.333, *p) / A
print("eff = %s" % eff)

latex.SI(eff, "", data_out, "messung1_eff")
latex.SI(A.format("3f"), "", data_out, "messung1_A")
latex.SI(x0*C.kilo, "\\nano\\second", data_out, "messung1_x0")
latex.SI(d*C.kilo, "\\nano\\second", data_out, "messung1_d")
latex.SI(y0, "", data_out, "messung1_y0")

xfit1 = np.linspace(wechselT, max(unv(dt)), num=250)
yfit1 = gaussUnc(xfit1, *p)
xfit2 = np.linspace(min(unv(dt)), wechselT, num=250)
yfit2 = gaussUnc(xfit2, *p)

print("A=%s\nx0=%s\nd=%s\ny0=%s" % (A,x0,d,y0))
sigma = 1
plt.fill_between(unv(xfit1), unv(yfit1) - sigma*usd(yfit1), unv(yfit1) + sigma*usd(yfit1), color="C1", alpha=0.4, zorder=8)
plt.plot(unv(xfit1), unv(yfit1), color="C1", alpha=1, linewidth=2, zorder=9, label="Gauss-Fit $\\pm %s\\sigma$" % sigma)
plt.fill_between(unv(xfit2), unv(yfit2) - sigma*usd(yfit2), unv(yfit2) + sigma*usd(yfit2), color="C1", alpha=0.4, zorder=8)
#plt.plot(unv(xfit2), unv(yfit2), color="C1", alpha=0.5, linestyle=":", linewidth=1, zorder=9, label="Gauss-Fit $\\pm %s\\sigma$" % sigma)

plt.xlabel(u"Reglerdifferenz $\\Delta t$ [$\\mu s$]")
plt.ylabel(u"Counts $N$")
#plt.grid()
#plt.xlim(0, 256)
#plt.ylim(0, 2000)
plt.legend(prop={'size':fig_legendsize}, loc="upper left")
plt.tick_params(labelsize=fig_labelsize, direction="in")
#plt.xscale('log')
#plt.yscale('log', nonposy='clip')
plt.savefig(data_out + "zeitdiferenz.png")
plt.savefig(data_out + "zeitdifferenz.pdf")
plt.show()
plt.close()

"""
- Halbwertszeit und Aktivität der Probe
- 
"""

# %% 2. Koinzidenzauflösung 

det1 = stat(624107) / (2*60) # s-1
det2 = stat(704247) / (2*60) # s-1
koinz = stat(4575) / (30*60) # s-1
zweiT = koinz / (det1 * det2) # s
print(zweiT/C.nano)
latex.SI(det1, "1\\per\\second", data_out, "messung2_det1")
latex.SI(det2, "1\\per\\second", data_out, "messung2_det2")
latex.SI(koinz, "1\\per\\second", data_out, "messung2_koinz")
latex.SI(zweiT/C.nano, "\\nano\\second", data_out, "messung2_zweiT")

halbwertszeit = unc.ufloat(30.08, 0.09)
aktiv = 100 * C.micro * 3.7e10 * 2**(np.log(0.5) * (2020 - 1962) / halbwertszeit) # Becquerel
print(2**(np.log(0.5) * (2020 - 1962) / halbwertszeit))
print(det1 / aktiv)
print(det2 / aktiv)

# %% 3. Vernichtungsstrahlung

det1 = stat(27134) / (2*60) # s-1
det2 = stat(30375) / (2*60) # s-1
koinz = det1 * det2 * zweiT # s-1
koinz90 = stat(557) / (10*60) - koinz # s-1
koinz120 = stat(486) / (10*60) - koinz #s-1
latex.SI(det1, "1\\per\\second", data_out, "messung3_det1")
latex.SI(det2, "1\\per\\second", data_out, "messung3_det2")
latex.SI(koinz, "1\\per\\second", data_out, "messung3_koinz")
latex.SI(koinz90, "1\\per\\second", data_out, "messung3_koinz90")
latex.SI(koinz120, "1\\per\\second", data_out, "messung3_koinz120")

print("Zufall: %s\n90 deg: %s\n120deg: %s" % (koinz, koinz90, koinz120))

# =============================================================================

def gauss(x, A, x0, sigma):
    return A * np.exp(-((x-x0)/sigma)**2/2)

def gaussUnc(x,A,x0,sigma):
    return A * unp.exp(-((x-x0)/sigma)**2/2)


fig = plt.Figure(figsize=fullscreen)

winkel, counts = messung3
counts -= koinz
plt.errorbar(unv(winkel), unv(counts), xerr=usd(winkel), yerr=usd(counts), fmt=" ", color="C0", zorder=10, label="Messung")

p0 = [22, 180, 5]
xfit, yfit, p = fit.fitspaceXY(winkel, counts, gauss, p0, range=(160,200), num=250, functionUnc=gaussUnc)
A,x0,d = p
winkelaufloesung = d
latex.SI(A, "1\\per\\second", data_out, "messung3_A")
latex.SI(x0, "\\degree", data_out, "messung3_x0")
latex.SI(d, "\\degree", data_out, "messung3_d")


print("A=%s\nx0=%s\nd=%s\ny0=%s" % (A,x0,d,y0))
sigma = 1
plt.fill_between(unv(xfit), unv(yfit) - sigma*usd(yfit), unv(yfit) + sigma*usd(yfit), color="C1", alpha=0.4, zorder=8)
plt.plot(unv(xfit), unv(yfit), color="C1", alpha=1, linewidth=2, zorder=9, label="Gauss-Fit $\\pm %s\\sigma$" % sigma)
#plt.fill_between(unv(xfit2), unv(yfit2) - sigma*usd(yfit2), unv(yfit2) + sigma*usd(yfit2), color="C1", alpha=0.4, zorder=8)
#plt.plot(unv(xfit2), unv(yfit2), color="C1", alpha=0.5, linestyle=":", linewidth=1, zorder=9, label="Gauss-Fit $\\pm %s\\sigma$" % sigma)

plt.xlabel(u"Winkel zwischen Detektoren $\\theta$ [$°$]")
plt.ylabel(u"Zählrate [$1/s$]")
#plt.grid()
#plt.xlim(0, 256)
#plt.ylim(0, 2000)
plt.legend(prop={'size':fig_legendsize}, loc="lower center")
plt.tick_params(labelsize=fig_labelsize, direction="in")
#plt.xscale('log')
#plt.yscale('log', nonposy='clip')
plt.savefig(data_out + "vernichtung.png")
plt.savefig(data_out + "vernichtung.pdf")
plt.show()
plt.close()

# %% 4. Winkelkorrelation

det1 = stat(71071) / (2*60) # s-1
det2 = stat(77452) / (2*60) # s-1
koniz = det1 * det2 * zweiT

zufall = stat(9) / (5*60) # s-1
count90 = stat(2198) / (30*60) #s-1
count180 = stat(2505) / (30*60) # s-1
print("count90 =  %s\ncount180 = %s\nkoinz =    %s\nzufall =   %s" % (count90, count180, koinz, zufall))

winkel = unp.uarray([90,180], unv(winkelaufloesung) + usd(winkelaufloesung))
counts = np.array([count90, count180]) - zufall

Aexp = (counts[1] - counts[0])/ (counts[0])
print("A_exp = %s" % Aexp)

def Wtheo(theta, A=1):
    return A * (1+1/8 * np.cos(theta)**2 + 1/24 * np.cos(theta)**4)

def WtheoUnc(theta, A=1):
    return A * (1+1/8 * unp.cos(theta)**2 + 1/24 * unp.cos(theta)**4
                )
Atheo = (Wtheo(180*C.grad) - Wtheo(90*C.grad)) / Wtheo(90*C.grad)
print("A_theo = %s" % Atheo)


latex.SI(det1, "1\\per\\second", data_out, "messung4_det1")
latex.SI(det2, "1\\per\\second", data_out, "messung4_det2")
latex.SI(koinz, "1\\per\\second", data_out, "messung4_koinz")
latex.SI(zufall, "1\\per\\second", data_out, "messung4_zufall")
latex.SI(count90, "1\\per\\second", data_out, "messung4_count90")
latex.SI(count180, "1\\per\\second", data_out, "messung4_count180")
latex.SI(Aexp.format(".3f"), "", data_out, "messung4_Aexp")
latex.SI("%.3f" % Atheo, "", data_out, "messung4_Atheo")

fig = plt.Figure(figsize=fullscreen)

plt.errorbar(unv(winkel), unv(counts), xerr=usd(winkel), yerr=usd(counts), fmt=" ", color="C0", zorder=10, label="Messung")

p0 = [2300]
xfit, yfit, p = fit.fitspaceXY(winkel, counts, Wtheo, p0, range=(0,2*math.pi), num=250, functionUnc=WtheoUnc)
xfit /= C.grad
A = p[0]
print("A=%s" % (A))
latex.SI(A, r"\per\second", data_out, "messung4_A")
sigma = 1
plt.fill_between(unv(xfit), unv(yfit) - sigma*usd(yfit), unv(yfit) + sigma*usd(yfit), color="C1", alpha=0.4, zorder=8)
plt.plot(unv(xfit), unv(yfit), color="C1", alpha=1, linewidth=2, zorder=9, label="Gauss-Fit $\\pm %s\\sigma$" % sigma)
#plt.fill_between(unv(xfit2), unv(yfit2) - sigma*usd(yfit2), unv(yfit2) + sigma*usd(yfit2), color="C1", alpha=0.4, zorder=8)
#plt.plot(unv(xfit2), unv(yfit2), color="C1", alpha=0.5, linestyle=":", linewidth=1, zorder=9, label="Gauss-Fit $\\pm %s\\sigma$" % sigma)

plt.xlabel(u"Winkel zwischen Detektoren $\\theta$ [$°$]")
plt.ylabel(u"Zählrate [$1/s$]")
#plt.grid()
#plt.xlim(0, 256)
#plt.ylim(0, 2000)
plt.legend(prop={'size':fig_legendsize})#, loc="lower center")
plt.tick_params(labelsize=fig_labelsize, direction="in")
#plt.xscale('log')
#plt.yscale('log', nonposy='clip')
plt.savefig(data_out + "theoKurve.png")
plt.savefig(data_out + "theoKurve.pdf")
plt.show()
plt.close()
