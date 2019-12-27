""" Berechne und plotte theoretische Kurvenverläufe """


""" Importanweisungen """
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

""" Konstanten für Absorbermaterial (Silizium) """
K = 0.1535 #MeV cm^2 / g
rho = 2.329#36  #g / cm^3 , bei 20°C, aus N. N. Greenwood, A. Earnshaw: Chemie der Elemente. 1988, ISBN 3-527-26169-9, S. 426.

Z = 14
A = 28.085

I = (9.76 + 58.8 * Z ** (-1.19)) * Z #eV
C_0 = -4.44
a = 0.1492
m_exp = 3.25
X_1 = 2.87
X_0 = 0.2014

dopRuhemElektron = 2 * 0.51099895 #MeV, aus CODATA Recommended Values. National Institute of Standards and Technology, abgerufen am 20. Mai 2019.


""" Grundlegende Formeln """

def gamma(E_kin, mc_squared):
    """ Berechnet gamma-Faktor, wenn E_kin und mc^2 in gleichen Einheiten """
    return 1 + E_kin / mc_squared

def beta(gamma_Wert):
    """ Berechnet beta aus gamma """
    return np.sqrt(1 - 1 / gamma_Wert ** 2)


""" Korrekturfaktoren """

def delta(gamma_Wert, C_0, a, m_exp, X_1, X_0):
    """ Berechnet Dichte-Korrekturfaktor """
    X = np.log10(gamma_Wert * beta(gamma_Wert))
    
    if X < X_0:
        
        return 0
    
    if X >= X_0 and X <= X_1:
        
        return 4.6052 * X + C_0 + a * (X_1 - X) ** m_exp
    
    if X > X_1:
        
        return 4.6052 * X + C_0


def C(gamma_Wert, I):
    """ Berechnet Dichte-Korrekturfaktor, wenn I in eV """
    eta = gamma_Wert * beta(gamma_Wert)
    
    if eta >= 0.1:
        return (0.422377 * eta ** (-2) + 0.0304043 * eta ** (-4) - 0.00038106 * eta ** (-6)) * 10 ** (-6) * I ** 2 + (3.850190 * eta ** (-2) - 0.1667989 * eta ** (-4) + 0.00157955 * eta ** (-6)) * 10 ** (-9) * I ** 3
    
    else: 
        
        return 0

""" Implementierung der Bethe-Bloch-Formel ohne z"""  

def BetheBloch_ohne_z(K, rho, Z, A, gamma_Wert, dopRuhemElektron, I, C_0, a, m_exp, X_1, X_0):
    """ Berechne rechte Seite der Bethe-Bloch-Formel geteilt durch z^2 """
    g = gamma_Wert
    b = beta(g)
    W_max = dopRuhemElektron * b * g
    
    return K * rho * Z / A / b ** 2 * (2 * np.log(W_max / I * 10 ** 6) - 2 * b ** 2 - delta(g, C_0, a, m_exp, X_1, X_0) - 2 * C(g, I) / Z)

""" Oftmals wird der spez. Energieverlust geplottet, d.h. teile durch die Dichte """

""" Erzeuge Datensatz aus Bethe-Bloch-Formel und interpoliere linear für bessere Laufzeit"""
fig = plt.figure()

eta = np.logspace(-3, 2, 100000)
eta_new = np.logspace(-3, 2, 1000000)

gamma_array = np.sqrt(1+eta**2)
gamma_new = np.sqrt(1+eta_new**2)
    
bethebloch = np.zeros((100000))

for i in range(100000):
    
    bethebloch[i] = BetheBloch_ohne_z(K, rho, Z, A, gamma_array[i], dopRuhemElektron, I, C_0, a, m_exp, X_1, X_0)
    
bethebloch_interpol = interp1d(gamma_array, bethebloch, kind='linear')

plt.xscale("log")
plt.yscale("log")
plt.plot(gamma_array, bethebloch, gamma_new, bethebloch_interpol(gamma_new),"--")
plt.show()
plt.clf()

""" Berechne Energieverlust in Detektor der Dicke d durch numerische Integration mit Riemann-Summe """


d = 8.7 * 10 ** (-4) #in cm, Wert von anderer Gruppe, ist anzupassen

N_steps = 100

Delta_x = d / N_steps ** 2

E_kin = np.linspace(.2, 20, N_steps) # MeV

Delta_E = np.zeros((N_steps))

E_kin_temp = 0.0

#bsp: alphas
z = 2
mc_squared = 3727.379 #MeV, Quelle: anderer Bericht S. 17

for i in range(N_steps):

    E_kin_temp = E_kin[i]
    
    x = 0
    while x <= d:
        
        x += Delta_x
        
        g = gamma(E_kin_temp, mc_squared)
        
        if g > np.sqrt(1 + 10e-6): #Interpolationsbereich
            
            E_kin_temp -= Delta_x * z ** 2 * bethebloch_interpol(g)
            
        else: 
            
            #Abbruch
            x = d + 1
    
    Delta_E[i] = E_kin[i] - E_kin_temp
    
plt.plot(E_kin, Delta_E)

plt.show()

