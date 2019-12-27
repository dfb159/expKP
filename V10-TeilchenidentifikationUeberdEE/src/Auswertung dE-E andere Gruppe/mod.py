import matplotlib.patches as mpatches
import numpy as np
import statistics as stat
import scipy as sci
import scipy.integrate as integrate
import scipy.fftpack
import sympy as sym
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.axes as axes
from matplotlib import colors as mcolors
import math
from scipy import optimize
import uncertainties as unc
import uncertainties.unumpy as unp
import uncertainties.umath as umath
import glob
import os
import matplotlib.pyplot as plt
from scipy.odr import *
from tqdm import tqdm
import matplotlib.pylab as pylab
import pathlib
import types
import inspect

#TODO create folders for file saves

# %% Konstanten fuer einheitliche Darstellung
unv=unp.nominal_values
usd=unp.std_devs
#
fig_size = (8, 6)
fig_legendsize = 14
fig_labelsize = 12 # ‘xx-small’, ‘x-small’, ‘small’, ‘medium’, ‘large’, ‘x-large’, ‘xx-large’.
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (8, 6),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
matplotlib.rcParams.update(params)
#matplotlib.rcParams.update({'font.size': fig_labelsize})

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
#colors
def frange(x, y, jump):
  while x < y:
    yield x
    x += jump
def pr(a,nnl=False):
    if nnl:
        print(a,end='')    
    else:
        print(a)
    return a
show_=True
def show():
    if show_:
        plt.show()
# mathe Funktionen
def poisson_dist(N):
    return unp.uarray(N,np.sqrt(N))
def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
def find_nearest(array, value):
    array[find_nearest_index(array,value)]
def normalize(ydata):
   return (ydata-np.amin(ydata))/(np.amax(ydata)-np.amin(ydata))
def novar_mean(f):
    return np.sum(f)/len(f)
def mean(n):
    # find the mean value and add uncertainties
    k = np.mean(n)
    err = stat.variance(unv(n))
    return unc.ufloat(unv(k), math.sqrt(usd(k)**2 + err))

def fft(y):
    N = len(y)
    fft = scipy.fftpack.fft(y)
    return 2 * abs(fft[:N//2]) / N

    # allgemeine Fitfunktionen
def const(x,m):
    '''m'''
    return (np.ones(np.shape(x))*m)
def linear(x,m): # lineare Funktion mit f(x) = m * x
    return(m*x)

def Gerade(x, a, b): # gerade mit = f(x) = m * x + b
    '''a*x+b'''
    return (a*x + b)

def cos_abs(x, a, f, phi):
    '''a*|cos(2*π*f*(x-phi))|'''
    return a * np.abs(unp.cos(2*np.pi*f*(x-phi)))

def cyclicOff(x, a, f, phi, offset):
    return cyclic(x, a, f, phi) + offset

def gauss(x, x0, A, d, y):
    '''$A\\cdot \\exp\\left(\\frac{-(x-x0)^2}{2d^2}\\right)+y$'''
    return A * unp.exp(-(x - x0)**2 / 2 / d**2) + y

def Two_Gauss(x, x0, A0, d0, x1, A1, d1,y):
    '''$A\\cdot \\exp\\left(\\frac{-(x-x0)^2}{2d^2}\\right)+y$'''
    #return A0 * np.exp(-(x - x0)**2 / 2 / d0**2) + y0 + A1 * np.exp(-(x - x1)**2 / 2 / d1**2) + y1
    return gauss(x,x0,A0,d0,y)+gauss(x,x1,A1,d1,0)
def Six_Gauss(x, x0, A0, d0, x1, A1, d1,x2, A2, d2,x3, A3, d3,x4, A4, d4, x5, A5, d5,y):
    '''$A\\cdot \\exp\\left(\\frac{-(x-x0)^2}{2d^2}\\right)+y$'''
    #return A0 * np.exp(-(x - x0)**2 / 2 / d0**2) + y0 + A1 * np.exp(-(x - x1)**2 / 2 / d1**2) + y1
    return gauss(x,x0,A0,d0,y)+gauss(x,x1,A1,d1,0)+gauss(x,x2,A2,d2,0)+gauss(x,x3,A3,d3,0)+gauss(x,x4,A4,d4,0)+gauss(x,x5,A5,d5,0)
def Two_Exp(x,A0,A1,l0,l1):
    '''A0*exp(-l0*x)+A1*exp(-l1*x)'''
    return exponential(x,-l0,A0)+exponential(x,-l1,A1)
def exponential(x, c, y0):
    return np.exp(c * x) * y0

# fittet ein dataset mit gegebenen x und y werten, eine funktion und ggf. anfangswerten und y-Fehler
# gibt die passenden parameter der funktion, sowie dessen unsicherheiten zurueck
#
# https://stackoverflow.com/questions/14581358/getting-standard-errors-on-fitted-parameters-using-the-optimize-leastsq-method-i#
# Updated on 4/6/2016
# User: https://stackoverflow.com/users/1476240/pedro-m-duarte
def fit_curvefit(datax, datay, function, p0=None, yerr=None, **kwargs):
    pfit, pcov = \
         optimize.curve_fit(function,datax,datay,p0=p0,\
                            sigma=yerr, epsfcn=0.0001, **kwargs, maxfev=1000000)
    error = []
    for i in range(len(pfit)):
        try:
          error.append(np.absolute(pcov[i][i])**0.5)
        except:
          error.append( 0.00 )
    pfit_curvefit = pfit
    perr_curvefit = np.array(error)
    return unp.uarray(pfit_curvefit, perr_curvefit)

def fit_curve(datax,datay,function,p0=None,yerr=None,xerr=None):
    model = Model(lambda p,x : function(x,*p))
    realdata = RealData(datax,datay,sy=yerr,sx=xerr)
    odr = ODR(realdata,model,beta0=p0)
    out = odr.run()
    return unp.uarray(out.beta,out.sd_beta)

def _data_split(datax,datay):
    x = unv(datax)
    y = unv(datay)
    xerr = usd(datax)
    yerr = usd(datay)
    xerr = xerr if np.any(np.abs(xerr)>0) else None
    yerr = yerr if np.any(np.abs(yerr)>0) else None
    return x,y,xerr,yerr
def data_split(datax,datay,frange=None):
    if frange is not None:
        return _data_split(datax[frange[0]:frange[1]],datay[frange[0]:frange[1]])
    else:
        return _data_split(datax,datay)

def plt_data(datax,datay,axis=("",""),label=None,fmt=None):
    x,y,xerr,yerr = data_split(datax,datay)
    if axis[0] != "":
        plt.xlabel(axis[0])
    if axis[1] != "":
        plt.ylabel(axis[1])
    if  xerr is None and yerr is None :
        if fmt is None:
            plt.plot(x,y, label=label)
        else:
            plt.plot(x,y, fmt, label=label)
    else:
        plt.errorbar(x,y,yerr=yerr,xerr=xerr,fmt=" ",capsize=5,label=label)
        
def _fit(datax,datay,function,p0=None,frange=None):
    x,y,xerr,yerr =data_split(datax,datay,frange)
    def tmp(*x):
        return unv(function(*x))
    if xerr is not None:
        fit = fit_curve(x,y,tmp,p0=p0,xerr=xerr,yerr=yerr)
    else:
        fit = fit_curvefit(x,y,tmp,p0=p0,yerr=yerr)
    return fit
def plt_fit(datax,datay,function,p0=None,units=None,frange=None,sigmas=1):
    x,y,xerr,yerr =data_split(datax,datay,frange)
    fit = _fit(datax,datay,function,p0,frange)
    xfit = np.linspace(unv(x[0]),unv(x[-1]),1000)
    l = function.__name__ + ": f(x)=" + function.__doc__
    for i in range(1,len(function.__code__.co_varnames)):
        l = l + "\n"
        l = l + str(function.__code__.co_varnames[i]) + "=%s"%(fit[i-1])
        if units is not None:
            l = l + " " + units[i-1]
    if sigmas>0:
        ll, = plt.plot(xfit,function(xfit,*unv(fit)),"-")
        yfit = function(xfit,*fit)
        plt.fill_between(xfit, unv(yfit)-sigmas*usd(yfit),unv(yfit)+sigmas*usd(yfit),alpha=0.4,label=l,color = ll.get_color())    
    else:
        l, = plt.plot(xfit,function(xfit,*unv(fit)),"-",label=l)
        if frange is not None:
            xfit = np.linspace(unv(datax[0]),unv(datax[-1]))
            plt.plot(xfit,unv(function(xfit,*fit)),"--",color=l.get_color())
    return fit

def fit_n_plt(datax,datay,function,p0=None,axis=("",""),label=None,units=None,sname=None,lpos=0,frange=None,sigmas=1):
    #x,y,xerr,yerr = data_split(datax,datay)
    iplot()
    plt_data(datax,datay,axis,label)
    fit = plt_fit(datax,datay,function,p0,units,frange=frange,sigmas=sigmas)
    splot(sname,lpos)
    return fit
def plt_func(func,start,end):
    xfit = np.linspace(start,end)
    plt.plot(xfit,func(xfit))
# %% Ouput
def gf(i):
    return "{0:." + str(i) + "g}"
def out_si(fn,s,u="",fmt="{}"):
    mkdirs(fn)
    file = open(fn,"w")
    file.write(si(s,u,fmt))
    print(fn,": ", fmt.format(s), u)
    file.close()

def out(fn,s):
    mkdirs(fn)
    file = open(fn,"w")
    file.write(("%s"%(s)).replace("/",""))
    print(fn,": ", "%s"%(s))
    file.close()
def si(s,u="",fmt="{}"):
    return "\\SI{%s}{%s}"%((fmt.format(s)).replace("/","").replace("(","").replace(")",""),u)

def out_si_line(fn,tab,skip=0):
    out_si_tab(fn,np.transpose([[t] for t in tab]),skip)
def out_si_tab(fn, tab,skip=0, fmt="{}"):
    mkdirs(fn)
    file = open(fn,"w")
    for i in range(len(tab)):
        for j in range(len(tab[i])):
            if(j!=0):
                file.write(pr("&",nnl=True))
            if(j>=skip):
                file.write(pr(si(tab[i][j],fmt=fmt),nnl=True))
            else:
                file.write(pr("%s"%(tab[i][j]),nnl=True))
        file.write(pr("\\\\\n",nnl=True))
    file.close()
def iplot(size=None): #init
    #fig = plt.figure(figsize=fig_size)
    if size==None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=size)
def mkdirs(fn):
    pathlib.Path(fn).parent.mkdir(parents=True, exist_ok=True)
def splot(sname=None,lpos=0): #save
    #plt.legend(prop={'size':fig_legendsize})
    
    plt.tight_layout()
    if lpos>=0:
        plt.legend(loc=lpos)
    plt.grid()
    if not sname==None:
        mkdirs(sname)
        plt.savefig(sname +".pdf")
    show()
    #plt.show()
def dump_vars(fd):
    key = ""
    gl = globals()
    for key in gl:
        smart_out(fd + key,gl[key])
    print(globals())
def iter(a):
    return zip(range(len(a)),a)
def files(folder,ending):
    r = []
    i=0
    for file in os.scandir(folder):
        if file.path.endswith(ending):
            r.append((i,os.path.splitext(os.path.basename(file.path))[0],file.path))
            i=i+1
    return r
def smart_out(fn,x):
    '''TODO'''
    out_si(fn,x)
    '''if isinstance(x,list):
        out_si_tab(fn,x)
    elif isinstance(x,types.FunctionType):
        out(fn,inspect.getsourcelines(x)[0])
    else:
        out_si(fn,x)
    '''
# usage zB:
# pfit, perr = fit_curvefit(unv(xdata), unv(ydata), gerade, yerr = usd(ydata), p0 = [1, 0])
# fuer eine gerade mit anfangswerten m = 1, b = 0

# weitere Werte, Konstanten
# Werte von https://physics.nist.gov/cuu/Constants/index.html[0]

c = 299792458 # m/s
k_B = unc.ufloat_fromstr("1.38064852(79)e-23") # J K-1 [0]
h = unc.ufloat_fromstr("4.135667662(25)e-15") # eV s [0]
h_bar = h/(2*np.pi) # J/s
r_e = unc.ufloat_fromstr("2.8179403227(19)e-15") # m [0]
R = unc.ufloat_fromstr("8.3144598(48)") # J mol-1 K-1 [0]
K = 273.15 # kelvin
g = 9.81 # m/s^2
rad = 360 / 2 / np.pi
grad = 1/rad
