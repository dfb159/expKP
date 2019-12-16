import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy.odr import Model, RealData, ODR
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as unv
from uncertainties.unumpy import std_devs as usd

def uncertain(array): # True, if ALL datapoints have uncertainties
    k = usd(array) == 0
    for i in k:
        if i:
            return False
    return True

# fittet ein dataset mit gegebenen x und y werten, eine funktion und ggf. anfangswerten und y-Fehler
# gibt die passenden parameter der funktion mit unsicherheiten zurueck
def fitY(datax, datay, function, p0=None, epsfcn=0.0001, maxfev=10000, **kwargs):
    yerr = usd(datay) if uncertain(datay) else None
    pfit, pcov = curve_fit(function, unv(datax), unv(datay), p0=p0, sigma=usd(yerr), epsfcn=epsfcn, maxfev=maxfev, **kwargs)
    perr = np.sqrt(np.diag(pcov))
    return unp.uarray(pfit, perr)

def fitXY(datax, datay, function, p0, **kwargs):
    model = Model(lambda p,x : function(x,*p))
    realdata = RealData(unv(datax),unv(datay),sx=usd(datax),sy=usd(datay))
    odr = ODR(realdata,model,beta0=p0, **kwargs)
    out = odr.run()
    return unp.uarray(out.beta,out.sd_beta)

def fitspaceY(datax, datay, function, p0=None, xfit=None, range=None, **kwargs):
    if range is None:
        range = min(unv(datax)), max(unv(datax))
    if xfit is None:
        xfit = np.linspace(*range, **kwargs)
    p = fitY(datax, datay, function, p0=p0, **kwargs)
    return xfit, function(xfit, *p), p

def fitspaceXY(datax, datay, function, p0, xfit=None, range=None, num=50, functionUnc=None, **kwargs):
    if range is None:
        range = min(unv(datax)), max(unv(datax))
    if xfit is None:
        xfit = np.linspace(*range, num)
    if functionUnc is None:
        functionUnc = function
    p = fitXY(datax, datay, function, p0=p0, **kwargs)
    return xfit, functionUnc(xfit, *p), p

def interpolate(datax, datay):
    return CubicSpline(datax, datay)

def interpolatespace(datax, datay, range=None, **kwargs):
    if range is None:
        range = min(unv(datax)), max(unv(datax))
    xfit = np.linspace(*range, **kwargs)
    return xfit, interpolate(datax, datay)(xfit)
