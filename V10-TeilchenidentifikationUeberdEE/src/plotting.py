# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from uncertainties.unumpy import nominal_values as unv
from uncertainties.unumpy import std_devs as usd


sidescreen = (8, 6)
fullscreen = (10,6)
widescreen = (16,6)
flatscreen = (10,4)

matplotlib.rcParams.update({"xtick.direction": "in", "ytick.direction": "in"})

"""
color
zorder
"""

def setup(figsize=None):
    fig = plt.Figure(figsize=figsize)
    
def error(x, y, color="C0", zorder=20, fmt=" ", bottom=None, top=None, left=None, right=None, **kwargs):
    assert len(x) == len(y)
    mask = np.repeat(True, len(x))
    if left is not None:
        mask &= x >= left
    if right is not None:
        mask &= x <= right
    if bottom is not None:
        mask &= y >= bottom
    if top is not None:
        mask &= y <= top
    x = x[mask]
    y = y[mask]
    plt.errorbar(unv(x), unv(y), xerr=usd(x) if any(usd(x)) else None, yerr=usd(y) if any(usd(y)) else None, fmt=fmt, color=color, zorder=zorder, **kwargs)
    
def fit(x, y, sigma=1, alpha=0.4, alphaData=0.7, label=None, color="C1", zorder=10, filltype="y", **kwargs):
    plt.plot(unv(x), unv(y), alpha=alphaData, label=label + r"$\pm %s\sigma$" % sigma if label else label, color=color, zorder=zorder, **kwargs)
    if filltype == "y":
        plt.fill_between(unv(x), unv(y) - sigma*usd(y), unv(y) + sigma*usd(y), alpha=alpha, color=color, zorder=zorder, **kwargs)
    if filltype == "x":
        plt.fill_betweenx(unv(y), unv(x) - sigma*usd(x), unv(x) + sigma*usd(x), alpha=alpha, color=color, zorder=zorder, **kwargs)

def params(xlabel=None, ylabel=None, xlim=None, ylim=None, grid=False, legend=True, legendloc="best", xscale=None, yscale=None, xscaleparams={}, yscaleparams={}):
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if grid:
        plt.grid()
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    if legend:
        plt.legend(loc=legendloc) # prop={'size':fig_legendsize},
    #plt.tick_params(direction="in") # labelsize=fig_labelsize, 
    if xscale:
        if xscale == "log" and "nonposx" not in xscaleparams:
            xscaleparams["nonposx"] = "clip"
        plt.xscale(xscale, **xscaleparams)
    if yscale:
        if yscale == "log" and "nonposy" not in yscaleparams:
            yscaleparams["nonposy"] = "clip"
        plt.yscale(yscale, **yscaleparams)
        
def save(path, formats=["png", "pdf"],transparent=False):
    for f in formats:
        plt.savefig(path + ".%s" % f,transparent=transparent)
        
def finish():
    plt.show()
    plt.close()