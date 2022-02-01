import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib import rc
from IPython.display import Image
import numpy as np
from astropy.io import fits as pyfits
from astropy.io import fits
from scipy.ndimage import gaussian_filter
import scipy
from scipy.optimize import minimize, rosen, rosen_der
from scipy.special import factorial
import scipy.integrate as integrate
from scipy.integrate import quad
from itertools import starmap
from scipy import optimize
import corner
import time
from mpl_toolkits import mplot3d
import healpy as hp
from scipy import nan
import math
import scipy as sp
import time


def nfw_profile_density(r, gamma = 1):
    #need to eventually normalize to local density (4 GeV/cm^2)
    #also normalize to fraction of DM in black holes
    r_s = 20 #kpc
    #make sure r is in kpc
    rho_0 = get_rho0(gamma_forrho0 = gamma) #normalized such that ρ = 0.4 GeV/cm3 at r = 8.25 kpc
    #rho_0 = 0.4
    rho = rho_0/(r/r_s)**(gamma)/((1+r/r_s)**(3-gamma)) # GeV/cm^3
    return rho

def get_rho0(gamma_forrho0 = 1):
    #calculating the initial rho_0
    r = 8.25 #kpc
    R_s = 20 #kpc
    rho_NFW = 0.4 #GeV/cm^3
    rho_0 = (r/R_s)**(gamma_forrho0)*rho_NFW*(1+r/R_s)**(3-gamma_forrho0)
    #print('rho_0: {}'.format(rho_0))
    #rho_0 = .257
    return rho_0

def i_hate_trig(theta, l):
    R_GC = 8.25 #kpc
    theta_radians = theta/180*np.pi

    r = np.sqrt(l**2+R_GC**2-2*l*R_GC*math.cos(theta_radians))

    #B = np.arcsin(l*np.sin(theta)/r)
    #phi = 180-theta-B
    
    return r
    
    
def range_over_l(theta, gamma_here = 1):
    l = np.linspace(1, 60, 10000) #in kpc
    r = i_hate_trig(theta, l)
    #print(r)

    density = nfw_profile_density(r, gamma = gamma_here)
    #print(density)

    
    return l, density


def get_long_lat(longlatboo = True):

    filelist = ['Bremss_00320087_E_50-814008_MeV_healpix_128.fits', 'Bremss_SL_Z6_R20_T100000_C5_E_50-814008_MeV_healpix_128.fits', 'pi0_Model_A_E_50-814008_MeV_healpix_128.fits', 'pi0_Model_F_E_50-814008_MeV_healpix_128.fits', 'ICS_Model_A_E_50-814008_MeV_healpix_128.fits', 'ICS_Model_F_E_50-814008_MeV_healpix_128.fits']
    hdu = fits.open(filelist[0])
    numpix = np.linspace(0, hdu[0].header['NPIX']-1, num = hdu[0].header['NPIX'])
    NSIDE = int(hdu[0].header['NSIDE'])
    degrees = hp.pix2ang(NSIDE, np.array(numpix,dtype=np.int), lonlat = longlatboo)

    #all indices within 20 deg
    vec = hp.ang2vec(np.pi/2, 0)
    ipix_disc20 = hp.query_disc(nside=NSIDE, vec=vec, radius=np.radians(20), inclusive = False)
    
    #all indices within 25 deg
    ipix_disc25 = hp.query_disc(nside=NSIDE, vec=vec, radius=np.radians(25), inclusive = False)
    
    #the l and b values we want:
    
    testl = degrees[0][ipix_disc25]
    testb = degrees[1][ipix_disc25]
    
    #need to find the values of testl that are greater than 200 and subtract 360 from them
    
    where_large = np.argwhere(testl > 200)
    testl[where_large] = testl[where_large]-360

    return np.array(testl), np.array(testb), ipix_disc20, ipix_disc25
    #return np.array(testb), np.array(testl)
    #return np.array(degrees[1]), np.array(degrees[0])


def get_j_factors(gam = 1):
    
    
    btest, ltest, indices20, indices = get_long_lat()

    thetas = np.sqrt(btest**2+ltest**2)
    
    integral = np.empty(thetas.shape)
    
    #ls are in kpc
    start = time.time()
    R_GC = 8.25 #kpc
    r_s = 20 #kpc
    rho_NFW = 0.4 #GeV/cm^3
    l = np.linspace(1, 60, 5000) #in kpc
    l1 = l**2+R_GC**2
    l2 = 2*l*R_GC
    l3 = l*3.086e21
    rho_0 = (R_GC/r_s)**(gam)*rho_NFW*(1+R_GC/r_s)**(3-gam)

    
    for i, theta in enumerate(thetas):
        #get the NFW density, and the lengths
        r = np.sqrt(l1-l2*math.cos(theta/180*np.pi))/r_s
        n_BH = rho_0/(r)**(gam)/((1+r)**(3-gam)) # GeV/cm^3 #*1e-4 #this also includes rho_0, can also use 1e-4 for PBH as DM 
        integral[i] = (np.trapz(n_BH**2, x = l3)) #trapezoidal rule, with n_Bh on y axis and the length on x
        #in GeV^2/cm^5?

    '''
    integral = []
    
    for theta in thetas:
        #get the NFW density, and the lengths
        ls, rho = range_over_l(theta, gamma_here = gam) #GeV/cm^3
        n_BH = rho #*1e-4 #this also includes rho_0, can also use 1e-4 for PBH as DM 
        ls_incm3 = ls*3.086e21
        integral.append(np.trapz(n_BH**2, x = ls_incm3)) #trapezoidal rule, with n_Bh on y axis and the length on x
        #in GeV^2/cm^5?
    end = time.time()
    print(integral[30])
    print('total time old:')
    print(end - start)
    '''
    
    
    
            
    return thetas, integral, indices20, indices



def make_image(integral):
    
    b, l = get_long_lat()
    hdu = fits.open('Bremss_00320087_E_50-814008_MeV_healpix_128.fits')
    numpix = np.linspace(0, hdu[0].header['NPIX']-1, num = hdu[0].header['NPIX'])
    NSIDE = int(hdu[0].header['NSIDE'])
    index = hp.ang2pix(NSIDE, l, b, lonlat = True)

    degrees = hp.pix2ang(NSIDE, np.array(numpix,dtype=np.int), lonlat = True)
    
    inner_20_pos = (np.sqrt((degrees[0])**2+degrees[1]**2)> 20)
    inner_20_neg = (np.sqrt((degrees[0]-360)**2+degrees[1]**2)> 20)
    inner_20 = np.logical_and(inner_20_pos, inner_20_neg)
    
    
    data4 = hdu[0].data
    test20 = np.copy(data4)
    test20[inner_20] = np.nan
    test20[index] = integral*1e-30
    #print(test20.shape)
    
    #print(b[np.where(integral == np.nanmax(integral))])
    #print(l[np.where(integral == np.nanmax(integral))])
        
    #hp.mollview(np.log10(np.log10(test20)), rot=(0,0,90))
    hp.mollview(np.log10(test20))
    
def darksusy(massparticle, highe, lowe, channel = 5, filenm = 'yield_DS_keith40.dat'):
    x = np.loadtxt(filenm, dtype=str).T
    
    energies = x[1].astype(np.float)
    yieldperann = x[2].astype(np.float)
    
    good_energies = np.where((energies <= highe) & (energies > lowe))
    #print('how many good energies: {}'.format(len(good_energies[0])))
    
    final_integral = np.trapz(yieldperann[good_energies], x = energies[good_energies])
    #print('integrated darksusy: {}'.format(final_integral))
    
    return final_integral #returns in counts per energy bin
    
    #Need to integrate over the energy bin that we are considering
    
def darksusy2(massparticle, energy, channel = 5, filenm = 'yield_DS_keith40.dat'):
    x = np.loadtxt(filenm, dtype=str).T
    energies = x[1].astype(np.float)
    yieldperann = x[2].astype(np.float)
    
    #get point based on energy
    
    interpolated_ds = log_interp1d2(energies, yieldperann)    
    try:
        interpolated_ds(energy)
    except ValueError:
        return 0
    return interpolated_ds(energy) #returns in per GeV

def log_interp1d2(xx, yy, kind='linear'):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp
    
def log_interp1d(xx, yy, kind='linear'):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind, fill_value="extrapolate")
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp
    
def get_dNdE(highenergy, lowenergy, massx = 40, sigmav = 1.4e-26, analyze_data = False, for_normals = False, energyhere = 1, darkmfilename = 'yield_DS_keith40.dat'):
    print(darkmfilename)
    thetas, integral, indices20, indices25 = get_j_factors(gam = 1.2) #in GeV^2/cm^5?, gamma 1.2
    #print('sigmav: {}'.format(sigmav))
    #print('mass: {} GeV'.format(massx))
    #now we need to add the integral back into the array where it was originally
    blank_array = np.empty(196608)
    blank_array[:] = 0
    
    count = 0
    blank_array[indices25] = integral
    
        
    if for_normals:
        dnde = darksusy2(massx, energyhere, channel = 5, filenm = darkmfilename) #per GeV per ann
        #print('dnde: {}'.format(dnde))
        #print('rho squared dl domega')


        #need to only return the inner 20 degrees
        new_arr = blank_array[indices20]
        #print(np.nansum(new_arr)/196608*4*np.pi)
        return thetas, new_arr*dnde*sigmav/massx**2/8/np.pi/1e3#*ann #units of per second per str per cm^2 per MeV

    if analyze_data:
        dnde = 1
    else:
        dnde = darksusy(massx, highenergy, lowenergy, channel = 5, filenm = darkmfilename) #per ann
        #print('dnde:')
        #print(dnde)
    
    
    return thetas, blank_array*dnde*sigmav/massx**2/8/np.pi#/5938#/1e3#/5910 #units of photons  per str per sec per cm^2 5910 for area
    #do I need the 1e3?

