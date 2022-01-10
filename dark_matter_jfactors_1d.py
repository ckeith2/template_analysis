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

def nfw_profile_density(r, gamma = 1):
    #need to eventually normalize to local density (4 GeV/cm^2)
    #also normalize to fraction of DM in black holes
    r_s = 20 #kpc
    #make sure r is in kpc
    #rho_0 = get_rho0(massBH, gamma_forrho0 = gamma)
    rho_0 = get_rho0() #GeV/cm^3
    rho = rho_0/(r/r_s)**(gamma)/((1+r/r_s)**(3-gamma)) #.4 GeV/cm^3
    return rho

def get_rho0(gamma_forrho0 = 1):
    #calculating the initial rho_0
    r = 8.25 #kpc
    R_s = 20 #kpc
    rho_NFW = 0.4 #BHs/cm^3, should be BHs/cm^3
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
'''
def get_long_lat(longlatboo = True):

    filelist = ['Bremss_00320087_E_50-814008_MeV_healpix_128.fits', 'Bremss_SL_Z6_R20_T100000_C5_E_50-814008_MeV_healpix_128.fits', 'pi0_Model_A_E_50-814008_MeV_healpix_128.fits', 'pi0_Model_F_E_50-814008_MeV_healpix_128.fits', 'ICS_Model_A_E_50-814008_MeV_healpix_128.fits', 'ICS_Model_F_E_50-814008_MeV_healpix_128.fits']
    hdu = fits.open(filelist[0])
    numpix = np.linspace(0, hdu[0].header['NPIX']-1, num = hdu[0].header['NPIX'])
    NSIDE = int(hdu[0].header['NSIDE'])
    degrees = hp.pix2ang(NSIDE, np.array(numpix,dtype=np.int), lonlat = longlatboo)
    
    #print(degrees[0]) #longitude l
    #print(degrees[1]) #latitude b
    degreesl = np.reshape(degrees[0], ((128*3, 1536//3)))
    degreesb = np.reshape(degrees[1], ((128*3, 1536//3)))
    
    inner_20_pos1 = np.argwhere(np.sqrt((degreesl)**2+degreesb**2)<= 20)
    inner_20_neg1 = np.argwhere(np.sqrt((degreesl-360)**2+(degreesb)**2)<= 20)
    inner_20_neg_tf1 = np.where(np.sqrt((degreesl-360)**2+(degreesb)**2)<= 20)
    #degreesl[inner_20_neg_tf] = degreesl[inner_20_neg_tf]-360
    inner_201 = np.concatenate((inner_20_pos1, inner_20_neg1))
    
    generous_ring = 25
    inner_20_pos = np.argwhere(np.sqrt((degreesl)**2+degreesb**2)<= generous_ring)
    inner_20_neg = np.argwhere(np.sqrt((degreesl-360)**2+(degreesb)**2)<= generous_ring)
    inner_20_neg_tf = np.where(np.sqrt((degreesl-360)**2+(degreesb)**2)<= generous_ring)
    #degreesl[inner_20_neg_tf] = degreesl[inner_20_neg_tf]-360
    inner_20 = np.concatenate((inner_20_pos, inner_20_neg))
    
    col1 = inner_201.T[0]
    row1 = inner_201.T[1]
    
    col = inner_20.T[0]
    row = inner_20.T[1]
    
    col_max = np.max(col)
    col_min = np.min(col)
    row_max = np.max(row)
    row_min = np.min(row)
    
    #degreesl[np.min(inner_20_neg.T[0])-1:np.max(inner_20_neg.T[0]),np.min(inner_20_neg.T[1]):np.max(inner_20_neg.T[1])+1] = degreesl[np.min(inner_20_neg.T[0])-1:np.max(inner_20_neg.T[0]),np.min(inner_20_neg.T[1]):np.max(inner_20_neg.T[1])+1] - 360
    
    degreesl[np.min(inner_20_neg.T[0])-1:np.max(inner_20_neg.T[0]),np.min(inner_20_neg.T[1]):np.max(inner_20_neg.T[1])+1] = degreesl[np.min(inner_20_neg.T[0])-1:np.max(inner_20_neg.T[0]),np.min(inner_20_neg.T[1]):np.max(inner_20_neg.T[1])+1] - 360

    
    testl = degreesl[col_min:col_max,row_min:row_max]
    testb = degreesb[col_min:col_max,row_min:row_max]
    #find lowest and highest of first column, lowest and highest of second column
    
    #create an empty array of the original 
    
    
    
    #trueb = list(degrees[1][inner_20_pos])+list(degrees[1][inner_20_neg])
    #truel = list(degrees[0][inner_20_pos])+list(degrees[0][inner_20_neg])
    return np.array(testl.flatten()), np.array(testb.flatten()), [(col_min, col_max), (row_min, row_max)], [col1, row1]
    #return np.array(testb), np.array(testl)
    #return np.array(degrees[1]), np.array(degrees[0])
'''

def get_j_factors(gam = 1):

    #btest, ltest, indices, indices20 = get_long_lat()
    btest, ltest = get_long_lat()

    thetas = np.sqrt(btest**2+ltest**2)

    integral = []
    
    #ls are in kpc
    
    for theta in thetas:
        #get the NFW density, and the lengths
        ls, rho = range_over_l(theta, gamma_here = gam)
        n_BH = rho #*1e-4 #this also includes rho_0, can also use 1e-4 for PBH as DM 
        ls_incm3 = ls*3.086e21
        integral.append(np.trapz(n_BH**2, x = ls_incm3)) #trapezoidal rule, with n_Bh on y axis and the length on x
    integral = np.asarray(integral) #in GeV/cm^2?
            
    #return thetas, integral, indices, indices20#*lum/4/np.pi #units of GeV/cm^-2? 
    return thetas, integral#*lum/4/np.pi #units of GeV/cm^-2? 

def get_long_lat(longlatboo = True):

    filelist = ['Bremss_00320087_E_50-814008_MeV_healpix_128.fits', 'Bremss_SL_Z6_R20_T100000_C5_E_50-814008_MeV_healpix_128.fits', 'pi0_Model_A_E_50-814008_MeV_healpix_128.fits', 'pi0_Model_F_E_50-814008_MeV_healpix_128.fits', 'ICS_Model_A_E_50-814008_MeV_healpix_128.fits', 'ICS_Model_F_E_50-814008_MeV_healpix_128.fits']
    hdu = fits.open(filelist[0])
    numpix = np.linspace(0, hdu[0].header['NPIX']-1, num = hdu[0].header['NPIX'])
    NSIDE = int(hdu[0].header['NSIDE'])
    degrees = hp.pix2ang(NSIDE, np.array(numpix,dtype=np.int), lonlat = longlatboo)
    
    #print(degrees[0]) #longitude l
    #print(degrees[1]) #latitude b
    
    degreesl = degrees[0]
    degreesb = degrees[1]
    
    inner_20_pos = np.where(np.sqrt((degreesl)**2+degreesb**2)<= 20)
    inner_20_neg = np.where(np.sqrt((degreesl-360)**2+(degreesb)**2)<= 20)
    
    #inner_20 = np.concatenate((inner_20_pos, inner_20_neg))
    
    testb = list(degreesb[inner_20_pos])+list(degreesb[inner_20_neg])
    testl = list(degreesl[inner_20_pos])+list((degreesl[inner_20_neg]-360))
    
    
    #return np.array(testl.flatten()), np.array(testb.flatten()), [(col_min, col_max), (row_min, row_max)], [col1, row1]
    return np.array(testb), np.array(testl)


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
    print(test20.shape)
    
    print(b[np.where(integral == np.nanmax(integral))])
    print(l[np.where(integral == np.nanmax(integral))])
        
    #hp.mollview(np.log10(np.log10(test20)), rot=(0,0,90))
    hp.mollview(np.log10(test20))
    
def darksusy(massparticle, highe, lowe, channel = 5):
    x = np.loadtxt('yield_DS_keith100.dat', dtype=str).T
    
    energies = x[1].astype(np.float)
    yieldperann = x[2].astype(np.float)
    
    good_energies = np.where((energies <= highe) & (energies > lowe))
    #print('how many good energies: {}'.format(len(good_energies[0])))
    
    final_integral = np.trapz(yieldperann[good_energies], x = energies[good_energies])
    #print('integrated darksusy: {}'.format(final_integral))
    
    return final_integral #returns in counts per energy bin
    
    #Need to integrate over the energy bin that we are considering
    
def get_dNdE(highenergy, lowenergy, massx = 100, sigmav = 2.2e-26, analyze_data = False, for_normals = False):
    thetas, integral = get_j_factors(gam = 1) #in GeV^2/cm^5?

    #make test image
    #make_image(integral)
    
    print('mass of dark matter: {}'.format(massx))
    if analyze_data:
        dnde = 1
    else:
        dnde = darksusy(massx, highenergy, lowenergy, channel = 5)
        sigmav = 2.2e-26
    #dnde = 1
    #print('dnde: {}'.format(dnde))
    #make_image(integral)
    return thetas, integral*dnde*sigmav/massx**2/8/np.pi/5910#/1e3#/5910 #units of photons  per str per sec per cm^2, 5910 for area
    #do I need the 1e3?
 

'''
def get_dNdE(highenergy, lowenergy, massx = 100, sigmav = 2.2e-26, analyze_data = False, for_normals = False):
    thetas, integral, indices, indices20 = get_j_factors(gam = 1) #in GeV^2/cm^5?

    
    #now we need to add the integral back into the array where it was originally
    blank_array = np.zeros((128*3, 1536//3))
    
    xshape = np.abs(indices[0][0]-indices[0][1])
    yshape = np.abs(indices[1][0]-indices[1][1])
    
    integral_reshaped = np.reshape(integral, (xshape, yshape))
    
    blank_array[indices[0][0]:indices[0][1], indices[1][0]:indices[1][1]] = integral_reshaped
    #blank_array[indices20[0], indices20[1]] = np.nan
    
    #plt.figure(figsize = (20,20))
    #plt.imshow(np.log10(blank_array))
    
    print('mass of dark matter: {}'.format(massx))
    if analyze_data:
        dnde = 1
    else:
        dnde = darksusy(massx, highenergy, lowenergy, channel = 5)
        sigmav = 2.2e-26
    #dnde = 1
    #print('dnde: {}'.format(dnde))
    #make_image(integral)
    if for_normals:
        #need to only return the inner 20 degrees
        new_arr = blank_array[indices20[0], indices20[1]]
        return thetas, new_arr*dnde*sigmav/massx**2/8/np.pi/5910
    return thetas, blank_array*dnde*sigmav/massx**2/8/np.pi/5910#/1e3#/5910 #units of photons  per str per sec per cm^2, 5910 for area
    #do I need the 1e3?
'''
