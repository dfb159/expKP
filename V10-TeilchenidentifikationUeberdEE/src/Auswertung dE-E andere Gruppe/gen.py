# %% Importanweisungen
import sys
sys.path.insert(1,"python_module/")
from mod import *
from par import *
unv=unp.nominal_values
usd=unp.std_devs
#compat
two_gauss=Two_Gauss
# Unsicherheiten

unc_n = 0
unc_p = 0
# %% Kali E
xscale=1
fname = "kaliohnealles"
peak1=unc.ufloat(5485.56,0.12)
peak2=unc.ufloat(5442.86,0.12)
for i in range(2):
    data = np.loadtxt("V10/raw/KaliOhneAlles/histograms_adc.dat")
    uxscale = unv(xscale)
    xdata = unp.uarray(data[:,0],0)*xscale
    ydata = unp.uarray(data[:,1],np.sqrt(data[:,1]))
    fit = fit_curvefit(unv(xdata),unv(ydata),two_gauss,p0=[1900*uxscale,800,20*uxscale,0,1800*uxscale,400,20*uxscale,0])
    #fit2 = fit_curve(unv(xdata),unv(ydata),two_gauss,p0=[1920*uxscale,800,30*uxscale,0,1790*uxscale,300,37*uxscale,0],yerr=usd(ydata),xerr=usd(xdata)if i==1 else None)
    #print("1 ", fit)
    #print("2 ", fit2)
    xfit = np.linspace(unv(xdata[0]),unv(xdata[-1]),1000)
    iplot()
    plt.errorbar(unv(xdata),unv(ydata),yerr=usd(ydata),xerr=usd(xdata),fmt=" ",capsize=5,label="measure")
    plt.plot(xfit,two_gauss(xfit,*unv(fit)),label="two Gaussians:\n%s $\\sigma$=%s\n%s $\\sigma$=%s"%(fit[0],fit[2],fit[4],fit[6]))
    #plt.plot(xfit,two_gauss(xfit,*unv(fit2)),label="two Gauß-Fits:\n%s $\\sigma$=%s\n%s $\\sigma$=%s"%(fit2[0],fit2[2],fit2[4],fit2[6]))
    plt.ylabel("count of events N")
    plt.xlabel("kinetic energy $E_{kin}$ in keV" if i else "channel")
    splot("V10/gen/"+fname+str(i))
    if i==0:
        xscale = peak1/unc.ufloat(unv(fit[0]),unv(fit[2]))

iplot()
plt.errorbar([0,unv(fit[4]/xscale),unv(fit[0]/xscale)],[0,unv(peak2),unv(peak1)],yerr=[0,usd(peak2),usd(peak1)],xerr=[0,unv(fit[6]/xscale),unv(fit[2]/xscale)],fmt=" ",capsize=5,label="peaks")
plt.plot([0,unv(fit[0]/xscale)],[0,unv(peak1)],label="calibration %s keV/ch"%(xscale))
plt.xlabel("channel")
plt.ylabel("energy E in keV" if i else "channel")
splot("V10/gen/kali_E")

# %% Test Fityk
fig = plt.figure(figsize=fig_size)
fname = "kaliohnealles"
data = np.loadtxt("V10/fit/" + fname + ".dat")
pdata = np.loadtxt("V10/fit/"+fname+".peaks", usecols=(2,3,4,5,6,7,8),skiprows=1)
plt.plot(data[:,0],data[:,1],'.')
plt.plot(data[:,0],data[:,2],label="Gaussian")
plt.plot(data[:,0],data[:,3],label="Gaussian")
plt.plot(data[:,0],data[:,4],label="Gaussian")
plt.ylabel("count of events N")
plt.xlabel("channel")
plt.grid()
plt.legend(prop={'size':fig_legendsize})
show()
# %% Kali  DeltaE
# TODO why does 15% peak disappear?
fname = "kaliundde"
data = np.loadtxt("V10/raw/KaliUndDE/histograms_adc.dat")
xdata = unp.uarray(data[:,0],0)*xscale
ydata = unp.uarray(data[:,1],np.sqrt(data[:,1]))
iplot()
plt.errorbar(unv(xdata),unv(ydata),yerr=usd(ydata),xerr=usd(xdata),fmt=" ",capsize=5,label="measure")
fit = fit_curvefit(unv(xdata),unv(ydata),gauss,p0=[4400,800,20,0])
xfit = np.linspace(unv(fit[0]-2*fit[2]),unv(fit[0]+2*fit[2]),1000)
plt.plot(xfit,gauss(xfit,*unv(fit)),label="Gaussian:\n%s $\\sigma$=%s"%(fit[0],fit[2]))
plt.ylabel("events N")
plt.xlabel("energy in keV")
splot("V10/gen/"+fname)
delta_e = peak1 - unc.ufloat(unv(fit[0]),unv(fit[2]))
xxscale = 1
for i in range(2):
    uxxscale = unv(xxscale)
    xdata = unp.uarray(data[:,0],0)*xxscale
    ydata =unp.uarray(data[:,2],np.sqrt(data[:,2]))
    iplot()
    plt.errorbar(unv(xdata),unv(ydata),yerr=usd(ydata),xerr=usd(xdata),fmt=" ",capsize=5,label="measure")
    fit = fit_curvefit(unv(xdata),unv(ydata),gauss,p0=[500*uxxscale,1200,20*uxxscale,0])
    xfit = np.linspace(unv(fit[0]-2*fit[2]),unv(fit[0]+2*fit[2]),1000)
    plt.plot(xfit,gauss(xfit,*unv(fit)),label="Gaussian:\n%s $\\sigma$=%s"%(fit[0],fit[2]))
    plt.ylabel("count of events N")
    plt.xlabel("energy loss $\Delta E$ in keV" if i else "channel")
    splot("V10/gen/"+fname + str(i))
    if not i:
        xxscale = delta_e/unc.ufloat(unv(fit[0]),unv(fit[2]))



# %% Mylars
fig = plt.figure(figsize=fig_size)
#iplot()
ax = fig.add_subplot(111)
ax.set_position([0.1,0.1,0.8,0.59])
i = -1
delta_esf = [0,0,0]
for fname in ["eineFolie","zweiFolien","dreiFolien"]:
    i+=1
    data = np.loadtxt("V10/raw/FolienDicke/" + fname +"/histograms_adc.dat")
    uxxscale = unv(xxscale)
    xdata = unp.uarray(data[:,0],0)*xscale
    ydata =unp.uarray(data[:,1],np.sqrt(data[:,1]))
    ax.errorbar(unv(xdata),unv(ydata),yerr=usd(ydata),xerr=usd(xdata),fmt=" ",capsize=5,label="measure " + str(i+1) +" foils")
    fit = fit_curvefit(unv(xdata),unv(ydata),two_gauss,p0=[4520-i*1255,700,100,0,4020-i*1255,300,100,0])
    xfit = np.linspace(unv(xdata[0]),unv(xdata[-1]),1000)
    delta_esf[i] = peak1-unc.ufloat(unv(fit[0]),unv(fit[2]))
    #xfit = np.linspace(unv(fit[0]-2*fit[2]),unv(fit[0]+2*fit[2]),1000)
    ax.plot(xfit,two_gauss(xfit,*unv(fit)),label="two Gaussians:\n%s keV $\\sigma$=%s keV\n%s keV $\\sigma$=%s keV"%(fit[0],fit[2],fit[4],fit[6]))
leg = ax.legend(prop={'size':11},loc = 'upper center', ncol=2, mode="expand", bbox_to_anchor = (0,1.03,1,.5))
plt.ylabel("count of events N")
plt.xlabel("kinetic energy $E_{kin}$ in keV" if i else "channel")
#plt.legend(prop={'size':fig_legendsize})
plt.grid()
plt.savefig("V10/gen/foiles.pdf")
    #show()
show()

# %% dE width
silicium_data = np.array([[4000,4500,5000,5500],[1.613e+2,1.505e+2,1.414e+2,1.334e+2], [1.323e-1,1.195e-1,1.091e-1,1.004e-1]])
mylar_data = np.array([  [1800,2000,2250,2500,2750,3000,3250,3500,3750,4000,4500,5000,5500],
                    [2.190e+2,2.080e+2,1.956e+2,1.846e+2,1.747e+2,1.658e+2,1.578e+2,1.505e+2,1.439e+2,1.379e+2,1.274e+2,1.184e+2,1.107e+2],
                    [2.006e-1,1.832e-1,1.654e-1,1.510e-1,1.390e-1,1.289e-1,1.202e-1,1.126e-1,1.060e-1,1.002e-1,9.039e-2,8.240e-2,7.577e-2]])
#data =mylar_data
#EF = EI-delta_esf[0]
def get_dist(steps):
    print(EF)
    dedx=data[1]+data[2]
    e = data[0]

    A = lambda j : (dedx[j+1]-dedx[j])/(e[j+1]-e[j])
    B = lambda j : (dedx[j]*e[j+1]-dedx[j+1]*e[j])/(e[j+1]-e[j])
    a = A
    b = B
    def sum_it(j,n):
        if j==n:
            return 0
        return sum_it(j+1,n)+1/a(j)*unp.log((a(j)*e[j+1]+b(j))/(A(j)*e[j]+B(j)))
    n=data.shape[1]-2
    end=n-steps
    #1/a(1)*unp.log((a(1)*e[2]+b(1))/(A(1)*e[1]+B(1)))
    d = 1/a(n)*unp.log((a(n)*EI+b(n))/(A(n)*e[n]+B(n)))+ sum_it(end+1,n)+ 1/a(end)*unp.log((a(end)*e[end+1]+b(end))/(A(end)*EF+B(end)))
    return d
EI = peak1
EF = EI-delta_e
data = silicium_data
silicium_thick = get_dist(2)
print("Silicium: ",silicium_thick)
silicium_thick = [silicium_thick,silicium_thick/1][0:1]
data =mylar_data
EF = EI-delta_esf[0]
foil1 = get_dist(2)
print("foil 1: ", foil1)
foil1=[foil1,foil1/1][0:1]

EF = EI-delta_esf[1]
foil2 =get_dist(5)
print("foil 2: ", foil2)
foil2= [foil2,foil2/2][0:1]
EF = EI-delta_esf[2]
foil3 =  get_dist(11)
print("foil 3: ",foil3)
foil3 = [foil3,foil3/3][0:1]
#names = ["whole thicknes","sinle thickness"]
out_si_tab("V10/res/t_thick",np.transpose([silicium_thick,foil1,foil2,foil3]))


# %% Bethe plot
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
plt.rcParams['image.cmap'] = 'hot' # use hot but make zero events white
viridis = cm.get_cmap('jet', 256)
newcolors = viridis(np.linspace(0, 1, 256))
nn = np.array([256/256, 256/256, 256/256, 1])
newcolors[:1, :] = nn
newcmp = ListedColormap(newcolors)

i = -1
data5 = [[],[]]
data6 = []
for fname in ["keineFolie",*["eineFolie"+str(j)+"grad" for j in range(0,60+15,15)],*["zweiFolien"+str(j)+"grad" for j in range(0,45+15,15)],*["zweiFolien"+str(j)+"grad" for j in range(0,15+15,15)]]:
    iplot()
    print(fname)
    i+=1
    data = np.loadtxt("V10/raw/EdEMessung/" + fname +"_adc.dat")
    data2 = np.loadtxt("V10/raw/EdEMessung/" + fname +"_energy.dat")
    data3 = np.loadtxt("V10/raw/EdEMessung/" + fname +"_energy2d.dat")
    #print(data2)
    uxxscale = unv(xxscale)
    idd = 2
    xdata = unp.uarray(data[:,0],0)*(xscale if idd==1 else xxscale)
    ydata =unp.uarray(data[:,idd],np.sqrt(data[:,idd]))
    plt.errorbar(unv(xdata),unv(ydata),yerr=usd(ydata),xerr=usd(xdata),fmt=" ",capsize=5,label="measure " + str(i+1) +" foils")
    #plt.show()
    plt.plot(data2[:,0],data2[:,idd],label="measure " + str(i+1) +" foils")
    splot()
    print(np.sum(data2[:,idd]), " vs ", np.sum(unv(ydata)))
    data4 = [[],[]]
    for (x,y,z) in data3:
        for i in range(int(z)):
            data4[0].append(x)
            data5[0].append(x)
            data4[1].append(y)
            data5[1].append(y)

    iplot()
    plt.hist2d(data4[0],data4[1],bins=100,cmap=newcmp)
    plt.colorbar()
    plt.ylabel("$\\Delta$E")
    plt.xlabel("E")
    splot()

# %% Bethe-Bloch
def bethe_bloch_iter(E,z,M,d,h=1e-9): # h in m
    cur_E = E
    #print(d, h, int(d/h),d/h)
    for i in range(0,int(d/h)):
        cur_E = cur_E-bethe_bloch(cur_E,z,M)*h
        #print(cur_E)
    return E-cur_E

def bethe_bloch(E,z,M):  # E,M in MeV , z in e
    N_a = 6.022e23 # mol**-1
    r_e = 2.817e-15 #m
    m_e = 0.511 # MeV
    c = 1#299792458 # m/s
    # sili
    rho = 2.3290 # g/cm**3
    Z = 14
    A = 28.085
    # rel calc
    p = np.sqrt(E**2-M**2*c**4)/c**2
    g = np.sqrt(1+(p/M)**2)
    b = np.sqrt(1-1/g**2)
    v = b * c
    #W = 2*m_e*c**2*(b*g)**2
    W = 2*m_e*c**2*(b*g)**2/(1+2*(m_e/M)*np.sqrt(1+(b*g)**2)+(m_e/M)**2)

    # fix
    I = (9.76+58.8*Z**(-1.19))*Z # eV
    #print(I)
    X0 = 0.2014
    X1 = 2.87
    m = 3.25
    a = 0.1492
    C0 = 4.44
    #I = 137 # eV
    X = np.log10(b*g)
    d = 0 if X<X0 else (4.6052*X+C0+a*(X1-X)**m) if X<X1 else 4.6052*X+C0 # unitless
    C = 0 #unitless
    e = b*g
    if b*g<0.1:
        C = 0
    else:
        C = (0.422377*e**-2+0.0304043*e**-4-0.00038106*e**-6)*1.e-6*I**2+(3.850190*I**-2-0.1667989*e**-4+0.00157955*e**-6)*1.e-9*I**3
        #raise ValueError("add C function")

    preprefac = 2*np.pi*N_a*r_e**2*m_e*c**2
    preprefac = 0.1535 # MeVcm**2/g
    prefac = preprefac*rho*Z/A*z**2/b**2
    return prefac* (np.log(2*m_e*g**2*v**2*W/(I/1000**2)**2)-2*b**2-d-2*C/Z) * 100 # MeV/m
print(bethe_bloch(0.5+3725,2,3725)*8.7*10**-6)
print("bethe-iter: ", bethe_bloch_iter(0.5+3725,2,3725,8.7e-6,8.7e-6))
# %% plot all
shift = 0.
#iplot((16,5))
iplot()
plt.hist2d(np.array(data5[0])+shift,np.array(data5[1])+shift,bins=99,cmap=newcmp)
xl=np.linspace(100,6000,500)
alpha = ["alpha",2,3727.379]
proton = ["proton",1,938.272]
deuteron = ["deuterium",1,1875.61]
tritium = ["tritium",1,2809.432]
helium3 = ["helium-3",2,2809.414]
lithium = ["lithium",3,6465.5]
all = [proton,deuteron,tritium,helium3,lithium,alpha]
out_si_tab("V10/res/t_part",all,skip=1)
for n,q,m in all:
    yl = []
    for x in tqdm(xl):
        #yl.append(-bethe_bloch(x/1000+3725,q,m)*8.7e-6*1000)
        yl.append(bethe_bloch_iter(x/1000+m,q,m,8.7e-6,1e-7)*1000)
    plt.plot(xl,yl,label=n)
#plt.plot(xl,yl,label="bethe")
#plt.gca().axvline(x=unv(peak1))
plt.colorbar()
plt.ylabel("energy loss $\\Delta$E in keV")
plt.xlabel("kinetic energy $E_{kin}$ in keV")
splot("V10/gen/bethe",2)


# %% Other pyplot
# Mass???
for n,q,m in [alpha,proton,deuteron,tritium,helium3,lithium]:
    data6 = []
    for k in range(len(data5[0])):
        x = data5[0][k]
        y = data5[1][k]
        p = np.sqrt((x/1000+m)**2-m**2)
        g = np.sqrt(1+(p/m)**2)
        b = np.sqrt(1-1/g**2)
        data6.append(b**2*y)
    d1 = []
    d2 = []
    for x in tqdm(xl):
            #yl.append(-bethe_bloch(x/1000+3725,q,m)*8.7e-6*1000)
            y = (bethe_bloch_iter(x/1000+m,q,m,8.7e-6,1e-7)*1000)
            p = np.sqrt((x/1000+m)**2-m**2)
            g = np.sqrt(1+(p/m)**2)
            b = np.sqrt(1-1/g**2)
            #d1.append(b**2*y)
            d1.append(x)
            d2.append(b**2*y)
    iplot()
    plt.plot(d1,d2,label=n)
    avg = np.average(d2[find_nearest_index(d1,3000):find_nearest_index(d1,5500)])
    splot()
    iplot()
    plt.hist(data6,bins=100,label=n)
    plt.grid()
    xll = np.linspace(0,10,100)

    plt.plot(xll,gauss(xll,avg,2500,0.6,0),label="Bethe-Bloch Gaussian")
    plt.ylabel("counts of events N")
    plt.xlabel("$\\Delta E\\beta^2$ in keV")
    splot("V10/gen/debb_"+n )
# %% All in one

xl=np.linspace(100,6500,500)
iplot()
for n,q,m in [alpha,proton,deuteron,tritium,helium3,lithium]:
    d1 = []
    d2 = []
    for x in tqdm(xl):
            #yl.append(-bethe_bloch(x/1000+3725,q,m)*8.7e-6*1000)
            y = (bethe_bloch_iter(x/1000+m,q,m,8.7e-6,1e-7)*1000)
            p = np.sqrt((x/1000+m)**2-m**2)
            g = np.sqrt(1+(p/m)**2)
            b = np.sqrt(1-1/g**2)
            #d1.append(b**2*y)
            d1.append(x)
            d2.append(b**2*y)
    plt.plot(d1,d2,label=n)
plt.xlabel("kinetic energy $E_{kin}$ in keV")
plt.ylabel("$\\Delta E\\beta^2$ in keV")
splot("V10/gen/debb")
#plt.hist(d1,bins=100)
