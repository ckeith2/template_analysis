import scipy as sp
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.interpolate

energies_low = [.3, .5, 1, 2, 5, 10]
energies_high = [10.001, 30, 50, 70, 100, 300, 500, 700, 1000, 3000] #in MeV
psf_low = [4.3, 2.5, 1.5, 1.1, 0.8, 0.8] #in degrees
psf_high = [9.5, 5.4, 2.7, 1.8, 1.3, 0.51, 0.3, 0.23, .15, .10]
effective_area_low = list(np.array([560, 446, 297, 117, 105, 50])*0.2*4*np.pi) #in cm^2*str
effective_area_high = list(np.array([215, 846, 1220, 1245, 1310, 1379, 1493, 1552, 1590, 1810])*.2*4*np.pi) #in cm^2*str


def log_interp1d(xx, yy, kind='linear'):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind, fill_value = 'extrapolate')
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

def get_acceptance_interp():
    
    return log_interp1d(energies_low+energies_high, effective_area_low+effective_area_high) #units of cm^2*str

def get_angle_interp():
    
    return log_interp1d(energies_low+energies_high, psf_low+psf_high) #units of degrees