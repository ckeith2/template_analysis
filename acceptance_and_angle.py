import scipy as sp
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.interpolate

location = './acceptance_angle/'

#angle and acceptance as a function of energy
csv_foracceptance = 'acceptance.csv'
csv_forangle = 'containment_angle.csv'

def log_interp1d(xx, yy, kind='linear'):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind, fill_value = 'extrapolate')
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

def get_acceptance_interp():
    accept = np.loadtxt(location + csv_foracceptance, delimiter = ',', dtype = float).T
    
    return log_interp1d(accept[0], accept[1]*1e4)

def get_angle_interp():
    angle = np.loadtxt(location + csv_forangle, delimiter = ',', dtype = float).T
    
    return log_interp1d(angle[0], angle[1])

    