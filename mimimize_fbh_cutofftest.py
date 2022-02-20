from iminuit import Minuit
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
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
import dark_matter_jfactors_test as dmj
import math
import random
import importlib
from pymultinest.solve import solve
import pymultinest
import warnings
warnings.filterwarnings("ignore")
import scipy as sp
import scipy.interpolate
import json
from os import walk
import re
import acceptance_psf_eastrogam as aaa
import photon_spectrum
import evaporating_black_hole_template as dm_template
import subprocess


def readfile(filename):
    """
    A simple function to read the maps of a given number n and given filename.
    """
    file_to_read = fits.open(filename)
    return file_to_read

def reshape_file(hdu, n, inner20 = True):
    """
    Reshapes the data to be in the size we want
    """
    hdu1 = readfile(filelist1[0])
    
    if inner20:
        numpix = np.linspace(0, hdu1[0].header['NPIX']-1, num = hdu1[0].header['NPIX'])
        NSIDE = int(hdu1[0].header['NSIDE'])
        degrees = hp.pix2ang(NSIDE, np.array(numpix,dtype=np.int), lonlat = True)
        vec = hp.ang2vec(np.pi/2, 0)
        ipix_disc20 = hp.query_disc(nside=NSIDE, vec=vec, radius=np.radians(20), inclusive = False)
        
        data4 = hdu[n].data
        test20 = np.copy(data4)[ipix_disc20]
        
        #might need this for plotting, not sure
        #data4 = hdu[n].data
        #test20 = np.copy(data4)
        #test20[inner_20] = np.nan
        #testbin = np.reshape(test20, (128*3, 1536//3))

    else:
        """
        testbin = np.reshape(hdu[n].data, (128*3, 1536//3))
        
        """
        numpix = np.linspace(0, hdu1[0].header['NPIX']-1, num = hdu1[0].header['NPIX'])
        NSIDE = int(hdu1[0].header['NSIDE'])
        degrees = hp.pix2ang(NSIDE, np.array(numpix,dtype=np.int), lonlat = True)
        
        inner_20_pos = (np.sqrt((degrees[0])**2+degrees[1]**2)> 20)
        inner_20_neg = (np.sqrt((degrees[0]-360)**2+degrees[1]**2)> 20)
        inner_20 = np.logical_and(inner_20_pos, inner_20_neg)
        
        data4 = hdu[n].data
        test20 = np.copy(data4)
        #test20[inner_20] = np.nan
        #print(sum(~np.isnan(test20)))
        #testbin = np.reshape(test20, (128*3, 1536//3))
        
    return test20

def get_energy_index(E_desired):
    idx = find_nearest(central_energies, E_desired)
    return idx
    
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    


def log_interp1d(xx, yy, kind='linear'):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind, fill_value="extrapolate")
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

def get_all_egb_data(energies, deltae):
    energy_range = np.array([(.1, .14), (.14, .2), (.2, .28), (.28, .4), (.4, .57), (.57, .8), (.8, 1.1), (1.1, 1.6), (1.6, 2.3), (2.3, 3.2), (3.2, 4.5), (4.5, 6.4), (6.4, 9.1), (9.1, 13), (13, 18), (18, 26), (26, 36), (36, 51), (51, 72), (72, 100), (100, 140), (140, 200), (200, 290), (290, 410), (410, 580), (580, 820)])*1e3 #GeV to MeV
    egb_intensity = np.array([3.7e-6, 2.3e-6, 1.5e-6, 9.7e-7, 6.7e-7, 4.9e-7, 3e-7, 1.8e-7, 1.1e-7, 6.9e-8, 4.2e-8, 2.6e-8, 1.7e-8, 1.2e-8, 6.8e-9, 4.4e-9, 2.7e-9, 1.8e-9, 1.1e-9, 6.2e-10, 3.1e-10, 1.9e-10, 8.9e-11, 6.3e-11, 2.1e-11, 9.7e-12])
    middle_bin = []
    bin_width = []
    for i in range(0, len(energy_range)):
        low_e = np.log10(energy_range[i][0])
        high_e = np.log10(energy_range[i][1])
        difference = np.abs((low_e+high_e)/2)
        middle_bin.append(10**(difference))
        bin_width.append(np.abs(energy_range[i][1]-(10**(difference)))) 
    return middle_bin, bin_width, egb_intensity

def get_all_egb(energies, deltae):
    energy_range = np.array([(.1, .14), (.14, .2), (.2, .28), (.28, .4), (.4, .57), (.57, .8), (.8, 1.1), (1.1, 1.6), (1.6, 2.3), (2.3, 3.2), (3.2, 4.5), (4.5, 6.4), (6.4, 9.1), (9.1, 13), (13, 18), (18, 26), (26, 36), (36, 51), (51, 72), (72, 100), (100, 140), (140, 200), (200, 290), (290, 410), (410, 580), (580, 820)])*1e3 #GeV to MeV
    egb_intensity = np.array([3.7e-6, 2.3e-6, 1.5e-6, 9.7e-7, 6.7e-7, 4.9e-7, 3e-7, 1.8e-7, 1.1e-7, 6.9e-8, 4.2e-8, 2.6e-8, 1.7e-8, 1.2e-8, 6.8e-9, 4.4e-9, 2.7e-9, 1.8e-9, 1.1e-9, 6.2e-10, 3.1e-10, 1.9e-10, 8.9e-11, 6.3e-11, 2.1e-11, 9.7e-12])
    middle_bin = []
    bin_width = []
    for i in range(0, len(energy_range)):
        low_e = np.log10(energy_range[i][0])
        high_e = np.log10(energy_range[i][1])
        difference = np.abs((low_e+high_e)/2)
        middle_bin.append(10**(difference))
        bin_width.append(np.abs(energy_range[i][1]-(10**(difference)))) 
        

    log_interp = log_interp1d(middle_bin, egb_intensity/bin_width, kind='linear')
    
    '''
    print(egb_intensity[2]/bin_width[2]*deltae[6])
    print(energies[6])
    x_trapz = np.logspace(np.log10(np.nanmin(energies)), np.log10(np.nanmax(energies)), num = 100)
    plt.scatter(middle_bin, egb_intensity/bin_width)
    plt.plot(x_trapz, log_interp(x_trapz), color = 'red')
    plt.scatter(energies[6], log_interp(energies[6]), color = 'green')
    plt.yscale('log')
    plt.xscale('log')
    '''
    
    counts = []
    #only want energies from 3 onward (about lowest at 80 MeV)
    for x in range(3, len(energies)):
        highest_val = energies[x]+deltae[x]
        lowest_val = energies[x]-deltae[x]
        x_trapz = np.logspace(np.log10(lowest_val), np.log10(highest_val), num = 400)
        #counts.append(log_interp(energies[x])*deltae[x])
        total_counts = np.trapz(log_interp(x_trapz), x = x_trapz)
        
        #print('total counts: {}'.format(total_counts))
        '''
        plt.scatter(middle_bin, egb_intensity/bin_width)
        plt.plot(x_trapz, log_interp(x_trapz), color = 'red')
        plt.scatter(energies[x], total_counts)
        
        plt.yscale('log')
        plt.xscale('log')
        asdfads
        '''
        counts.append(total_counts)  
    return counts #returns counts per cm^2 per sec per str

def subtract(n):
    ##Template for 1 GeV and 10 GeV
    icsa = readfile(filelist[n])
    
    idx1 = get_energy_index(1*1e3, icsa)
    idx10 = get_energy_index(10*1e3, icsa)
    
    array1 = reshape_file(icsa, idx1, inner20 = False)
    array10 = reshape_file(icsa, idx10, inner20 = False)
    
    ##Sum up idx1 and idx 10, make them equal in sum
    sum1 = np.nansum(array1)
    sum10 = np.nansum(array10)
    array10_adjusted = (array10*sum1/sum10)
    subtract110 = np.abs(array1-array10_adjusted)/array1
    fig, ax = plt.subplots(figsize=(10, 10))
    image = ax.imshow(subtract110)
    fig.colorbar(image, ax=ax, anchor=(0, 0.3), shrink=0.7)
    plt.title(str(filelist[n]))
    
    return



#n = 4 for ICSA, n = 2 for pi0
def psf_smoothing(n, energyidx, inner20psf = False, pointsource = False, use_og = False):
    inner20psf = False
    if pointsource:
        icsa = readfile(point_sources[0])
        data50 = icsa[energyidx].data
    else:
        icsa = readfile(filelist[n])
        data50 = reshape_file(icsa, energyidx, inner20 = inner20psf)
    hdu = readfile(filelist1[0])

    
    #get_where_within_20deg
    numpix = np.linspace(0, hdu[0].header['NPIX']-1, num = hdu[0].header['NPIX'])
    NSIDE = int(hdu[0].header['NSIDE'])
    vec = hp.ang2vec(np.pi/2, 0)
    ipix_disc = hp.query_disc(nside=NSIDE, vec=vec, radius=np.radians(20), inclusive = False)
    
    init_sum = np.sum(data50[ipix_disc])
    #print(init_sum)
    energy_here = float(central_energies[energyidx])
    sig = np.pi/180*angle_interp(energy_here)
    data50_convolved = hp.sphtfunc.smoothing(data50.flatten(), sigma=sig)

    fin_sum = np.sum(data50_convolved[ipix_disc])

    testbin = data50_convolved[ipix_disc]


    return np.array(testbin)*init_sum/fin_sum

def psf_smoothing_DM(energyidx, crosssec, anal_data = False, mass_dm = 40, use_og = False, evapbh = False, gam = 1, massbh = 2e16, lum_interp = 1, filename = 'yield_DS_keith40.dat'):
    energybin = central_energies[energyidx]
    bins_in_lin = np.log10(energybin)
    deltae = get_deltaE(energyidx)
    
    highe = (energybin+deltae)/1e3
    lowe = (energybin-deltae)/1e3
    
    hdu = readfile(filelist1[0])
    numpix = np.linspace(0, hdu[0].header['NPIX']-1, num = hdu[0].header['NPIX'])
    NSIDE = int(hdu[0].header['NSIDE'])
    
    degrees = hp.pix2ang(NSIDE, np.array(numpix,dtype=np.int), lonlat = True)

    

    #need to make sure the initsum is *only within the inner 20 degrees, same for finsum
    if evapbh == True:
        data50 = dm_template.get_dNdE(egamma_values, energyidx, lum_interp, gamma = gam, mass = massbh, for_normals = False)
        #photons per cm^2 per sec per str
    else:
        data50 = dmj.get_dNdE(highe, lowe, sigmav = crosssec, analyze_data = anal_data, massx = mass_dm, darkmfilename = filename)[1] #photons per cm^2 per sec per str

    
    #get_where_within_20deg
    numpix = np.linspace(0, hdu[0].header['NPIX']-1, num = hdu[0].header['NPIX'])
    NSIDE = int(hdu[0].header['NSIDE'])
    vec = hp.ang2vec(np.pi/2, 0)
    ipix_disc = hp.query_disc(nside=NSIDE, vec=vec, radius=np.radians(20), inclusive = False)
    
    init_sum = np.sum(data50[ipix_disc])
    #print('init sum: {}'.format(init_sum))
    
    energy_here = central_energies[energyidx]
    sig = np.pi/180*angle_interp(energy_here)
    if use_og:
        data50_convolved = hp.sphtfunc.smoothing(data50.flatten(), sigma=np.pi/180/1.508)
    else:
        data50_convolved = hp.sphtfunc.smoothing(data50.flatten(), sigma=sig)
    fin_sum = np.sum(data50_convolved[ipix_disc])
    
    #hp.mollview((data50_convolved), coord = 'G')
    
    #photons per cm^2 per sec per str

    
    return np.array(data50_convolved[ipix_disc])*init_sum/fin_sum

def get_deltaE(n):
    energybins = central_energies
    bins_in_lin = np.log10(energybins)[n]
    #spacing = 0.05691431 #old spacing
    spacing = 0.17609125905568124 #new spacing
    
    high_bin = 10**(bins_in_lin + spacing)
    low_bin = 10**(bins_in_lin - spacing)
    
    deltaE = np.abs(high_bin - low_bin)
    #print('delta E: {}'.format(deltaE))
    
    return deltaE



def poisson_dist(n, energyidx, cross_section =1.4e-26, dm = False, dm_bh = False, analyze_data = False, dm_mass = 40, egb = False, points = False, counts = 0, evapbh = True, blackholem = 2e16,  luminterpolated = 1, gamm = 1.6, filenm = 'yield_DS_keith40.dat'):   
    '''
    Performs a PSF smoothing of the array, before converting it into photons per pixel
    
    '''
    deltaE = get_deltaE(energyidx)
    #print(deltaE)
    energy_here = central_energies[energyidx]
    #print(energy_here)
    acceptance_for_poisson = acceptance_interp(energy_here) #in m^2*str, convert to cm^2
    if dm:
        convolved_data = psf_smoothing_DM(energyidx, cross_section, anal_data = analyze_data, mass_dm = dm_mass, filename = filenm)/deltaE
    elif egb:
        convolved_data_init = np.empty(5938) #needs to be the length of the good vals
        convolved_data_init.fill(1) #counts per cm^2 per sec per str
        convolved_data = convolved_data_init*counts #in units of photons per cm^2 per mev per str per sec
    elif points:
        convolved_data = psf_smoothing(n, energyidx, pointsource = True)
    elif dm_bh:
        convolved_data = psf_smoothing_DM(energyidx, cross_section, lum_interp = luminterpolated, anal_data = analyze_data, evapbh = True, massbh = blackholem, gam = gamm)/deltaE
    else:
        convolved_data = psf_smoothing(n, energyidx) #data in units of photons cm^-2 MeV^-1 str^-1
    #n_gamma = np.array(convolved_data)*deltaE*exposure_time*8500*4*np.pi/196608*.2 #13 years*.85meters^2, units of photons per pixel
    n_gamma = np.array(convolved_data)*deltaE*exposure_time*acceptance_for_poisson/196608 #13 years*.85meters^2, units of photons per pixel

    
    return n_gamma





def get_image(data):
    fig, ax = plt.subplots(figsize=(10, 10))
    image = ax.imshow(data)
    fig.colorbar(image, ax=ax, anchor=(0, 0.3), shrink=0.7)
    return

def simulated_data(energyidx, templates):
    
    '''
    Use PSF smoothed data to create a random poisson draw to obtain simulated data
    '''
    
    
    n_gammatot = 0
    for i in range(0, len(templates)):
        n_gammatot += templates[i]
    len_of_rand = len(n_gammatot)
    simdata = np.zeros(len_of_rand)
    randdata = np.random.rand(len_of_rand)
    
    
    #print(n_gammatot)
    
    for i in range(0, len_of_rand):
        #print(n_gammatot[i])
        if n_gammatot[i]<.01:
            if n_gammatot[i] < randdata[i]:
                simdata[i] = 0
            else:
                simdata[i] = 1
        else:
            simdata[i] = np.random.poisson(lam = n_gammatot[i])
    #print('simdata just 20: ', np.nansum(simdata))
    
    return simdata

def minimize_likelihood(energyidx, cross_sec = 1.4e-26, massdm = 40):
    pi = poisson_dist(2, energyidx)
    ics = poisson_dist(4, energyidx)   
    brem = poisson_dist(0, energyidx)

    darkm = poisson_dist(np.nan, energyidx, cross_section = cross_sec, dm = True, dm_mass = massdm)

    k = simulated_data(energyidx, [pi, ics, brem])#remove DM for accurate
    #print(np.nanmean(lamb))
    #asdfasd
    
    #Need to minimize for lamb < 50 and lamb > 50
    
    bnds = ((0, np.inf), (0, np.inf), (0, np.inf), (1e-30, np.inf))
    result = minimize(likelihood, (1, 1, 1, 1), args = (k, pi, ics, brem, darkm), bounds = bnds)
    min_likelihood = result.x
    chi2 = result.fun
    #print(result)

    return min_likelihood


def likelihood(constants, k, pi, ics, brem, dm):
    """
    Gets the Total Likelihoods from the Gaussian Regime and the Poisson Regime
    Once each has been calculated, multiplies values together for final likelihood
    """
    
    likelihood_poiss = likelihood_poisson(constants, k, pi, ics, brem, dm)
    #print('likelihood: {}'.format(likelihood_poiss)) 
    return likelihood_poiss

def merge(list1, list2):
      
    merged_list = tuple(zip(list1, list2)) 
    return merged_list

def likelihood_gaussian(constants, lamb, pi, ics):
    sigma = np.sqrt(constants[0]*pi+constants[1]*ics)
    mu = lamb
    rng = 0.5
    
    #flatten arrays
    sigma_flat = sigma.flatten()
    mu_flat = mu.flatten()
    length_flattened = len(mu_flat)
    #get arrays in sigma, mu tuple format
    ms_tuples = list(merge(mu_flat, sigma_flat))
    ms = np.array(ms_tuples, dtype = 'f,f')
    lower_bound_arr = mu_flat - rng
    upper_bound_arr = mu_flat + rng
    
    args = np.concatenate((np.full((length_flattened, 1), prob_func), lower_bound_arr.reshape((length_flattened, 1)), upper_bound_arr.reshape((length_flattened, 1)), ms.reshape((length_flattened, 1))), axis = 1)
    
    #log likelihood
    prob = list(starmap(lambda a, b, c, d: quad(a, b, c, d)[0], args))

    #reshape for testing
    l = np.sum(np.log(prob))
    likely = -2*l
    
    return likely
    
def prob_func(x, mu, sigma):
    probdens = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((x-sigma**2)/sigma)**2)
    return probdens

#def likelihood_poisson(a0):
def likelihood_poisson(a0, a1, a2, a3, a4):
#def likelihood_poisson(constants, ktest, pitest, icstest, bremtest, dmtest):
    lamb = a0*pitest+a1*icstest+a2*bremtest+a3*darkmtest+a4*egbtest
    #lamb = a0*egbtest
    
    #print(a0, a1, a2, a3)
    #print(lamb)

    #lamb = constants[0]*pitest+constants[1]*icstest+constants[2]*bremtest+constants[3]*dmtest
    
    fprob = -scipy.special.gammaln(ktest+1)+ktest*np.log(lamb)-lamb #log likelihood of poisson
    #print(fprob)
    return -2*np.nansum(fprob)

def get_curves_norm(n, energyidx, inner20psf = True):
    icsa = readfile(filelist[n])
    data50 = reshape_file(icsa, energyidx, inner20 = inner20psf) #get the data at energyidx MeV
    delt = get_deltaE(energyidx)
    
    hdu = readfile(filelist[0])
    energy_here = central_energies[energyidx]
    acceptance_forpoisson = acceptance_interp(energy_here) #in m^2*str, convert to cm^2
    
    return np.asarray(data50)*delt
    #return np.asarray(data50)*exposure_time*acceptance_forpoisson*delt #try to get in terms of counts per bin

def get_curves(n, energyidx, inner20psf = True):
    icsa = readfile(filelist[n])
    data50 = reshape_file(icsa, energyidx, inner20 = inner20psf) #get the data at energyidx MeV
    delt = get_deltaE(energyidx)
    
    hdu = readfile(filelist[0])
    energy_here = central_energies[energyidx]
    acceptance_forpoisson = acceptance_interp(energy_here) #in m^2*str, convert to cm^2
    
    return np.asarray(data50)*exposure_time*acceptance_forpoisson*delt #try to get in terms of counts per bin
    
    
def get_normalizations_spectrum(deltaE, cross_sec = 1.4e-26, dm_mass = 40, f_bh = 1, gam = 1, massbh = 2e26, darkmfile = 'yield_DS_keith40.dat'):
    
    range_templates = [2, 4, 0] #pi, ics, brem
    energies = np.copy(central_energies)
    templates = []
    temp = []
    #deltaomega = 1/196608 #i think this should maybe still be total bins in whole image pre-20?
    deltaomega=1
    acceptances = []
    for energy_here in range(0, len(energies)):
        acceptance_forpoisson = acceptance_interp(energy_here) #in cm^2*str
        acceptances.append(acceptance_forpoisson)
    

    for n in range_templates:
        temp = []
        for index in range(0, len(energies)):
            temp.append(np.nansum(get_curves_norm(n, index))) #units of counts per cm^2 per sec per sr
        templates.append(np.asarray(temp)*deltaomega) #gets counts per bin
    print('finished new templates!!')
    
    #eavporating black holes
    dmevap_temp = []
    dmevap_tot = []
    energiesforBH = np.logspace(np.log10(.05), np.log10(1e7), num = 1000) #in MeV
    #eventually make it so this is only calculated once per mass_bh
    lum = (photon_spectrum.get_integral(egamma_values, mass_bh = massbh)[1]) #units of photons per MeV per sec per BH
    #interpolate the luminosity
    lum_interp = log_interp1d(energiesforBH, lum, kind='linear') #integrate dn/dE in units of MeV
    
    for index in range(0, len(energies)):
        #print(index)
        energybin = energies[index]
        bins_in_lin = np.log10(energybin)
    
    
        data50 = dm_template.get_dNdE(egamma_values, index, lum_interp, gamma = gam, mass = massbh, for_normals = True) #units of photons cm^-2 str^-1 per sec
        #need to cut this to be inner 20 degrees
        
        
        #dm_temp.append(np.nansum(data50))
        dmevap_temp.append(np.nansum(data50)*4*np.pi)#units of photons cm^-2 per sec
        #print(np.asarray(dmevap_temp)/.38*f_bh/196608)
        #print('--------------------------------------')
    dmevap_tot.append(np.asarray(dmevap_temp)/.38*f_bh/196608) #photons per cm^2 per sec per sr

        
    #dark matter template

    dm_temp = []
    dm_templates_tot = []
    for index in range(0, len(energies)):
        #print(index)
        energybin = energies[index]
        bins_in_lin = np.log10(energybin)
        deltae = get_deltaE(index)
    
        highe = (energybin+deltae)/1e3
        lowe = (energybin-deltae)/1e3

        data50 = dmj.get_dNdE(highe, lowe, sigmav = cross_sec, massx = dm_mass, for_normals = True, energyhere = energybin/1e3, darkmfilename = darkmfile)[1]*deltae #units of photons cm^-2 str^-1 per sec
        #need to cut this to be inner 20 degrees
        
        #dm_temp.append(np.nansum(data50))
        dm_temp.append(np.nansum(data50)*4*np.pi) #units of photons cm^-2 per sec
        #print(np.asarray(dm_temp)/.38/196608)
        #print('--------------------------------------')
    dm_templates_tot.append(np.asarray(dm_temp)/.38/196608) #photons per cm^2 per sec per sr, .38 is delta omega
    
    print('DONE WITH  THE DARK MATTER TEMPLATE')
    

    
    #return range_templates, energies, [np.array(templates[0]), np.array(templates[1]), np.array(templates[2]), np.array(dmevap_tot)]

    
    
    #EGB template
    egb_templates = np.array(get_all_egb(energies, deltaE)) #units of counts per cm^2 per sec per str
    
    #Point Source Template
    point_source_arr = []
    for index in range(0, len(energies)):
        smaller_index = index
        point_source_arr.append(np.nansum(get_curves_pointsource(index, smaller_index, inner20psf = True))) ##units of photons per cm^2 per sec per sr
    
    
    print('yay!')
        
    #return range_templates, energies, [np.array(templates[0]), np.array(templates[1]), np.array(templates[2]), np.array(egb_temp_fin), np.array(dm_templates_tot), np.array(dmevap_tot)] #counts per pixel
    #return range_templates, energies, [np.array(templates[0]), np.array(templates[1]), np.array(templates[2]), np.array(egb_templates), np.array(point_source_arr), np.array(dm_templates_tot)]

    return range_templates, energies, [np.array(templates[0]), np.array(templates[1]), np.array(templates[2]), np.array(egb_templates), np.array(point_source_arr), np.array(dm_templates_tot), np.array(dmevap_tot)] #counts per pixel
    
def get_curves_pointsource(energyidx, smallindex, inner20psf = True):
    pointsourcedata = readfile(point_sources[0])[smallindex].data
    
    hdu = readfile(filelist1[0])
    numpix = np.linspace(0, hdu[0].header['NPIX']-1, num = hdu[0].header['NPIX'])
    NSIDE = int(hdu[0].header['NSIDE'])
    
    degrees = hp.pix2ang(NSIDE, np.array(numpix,dtype=np.int), lonlat = True)
    vec = hp.ang2vec(np.pi/2, 0)
    ipix_disc20 = hp.query_disc(nside=NSIDE, vec=vec, radius=np.radians(20), inclusive = False)
        
    test20 = np.copy(pointsourcedata)[ipix_disc20]
    
    delt = get_deltaE(energyidx)
    
    deltaE = get_deltaE(energyidx)
    hdu = readfile(filelist[0])
    energy_here = central_energies[energyidx]
    acceptance_forpoisson = acceptance_interp(energy_here) #in m^2*str, convert to cm^2
    
    return np.asarray(test20)*delt #units of photons per cm^2 per sec per sr
    
    #return np.asarray(test20)*exposure_time*acceptance_forpoisson*delt #13 years * .85 m^2 * .2, return in photons /bin
    
    
def get_normalized(energyidx, normals, template_val, energies):
    '''
    Normalizes the ROI based on the shape the spectrums should have
    
    
    Do not need to use this, as long as you stay consistent across all Fermi data
    for the exposure time and collecting area
    
    '''
    
    poisson_pi = poisson_dist(template_val, int(energyidx)) #units of photons per pixel
    init_sum_pi = np.nansum(poisson_pi)
    #print(np.nansum(init_sum_pi))
    if template_val == 2:
        normval = 0
    if template_val == 4:
        normval = 1
    if template_val == 0:
        normval = 2
    print('normval for pi at 0: {}'.format(normals[normval][energyidx])) 
    print('delta E at 0: {}'.format(get_deltaE(energyidx)))
    normal_pi = normals[normval][energyidx]*get_deltaE(energyidx)
    pitest = poisson_pi*normal_pi/init_sum_pi
    print('normalization: {}'.format(np.nansum(pitest)))
    #print(np.nansum(pitest))
    
    #print('----------------------')
    
    return pitest
    


def get_darksusy_counts():
    x = np.loadtxt('yield_DS_keith40.dat', dtype=str).T
    energies = x[1].astype(np.float)*1e3
    yieldperann = x[2].astype(np.float)/1e3 #convert from per GeV to per MeV
    energybins = np.copy(central_energies)
    
    counts = []
    delta = []
    for n in range(0, len(energybins)):
        
        bins_in_lin = energybins[n]
        deltae = get_deltaE(n)
    
        highe = (bins_in_lin+deltae)
        lowe = (bins_in_lin-deltae)
        
        good_energies = np.where((energies <=highe) & (energies >= lowe))
        
        final_integral = np.trapz(yieldperann[good_energies], x = energies[good_energies])

        counts.append(final_integral)
        delta.append(deltae)
    return np.array(counts), np.array(delta)



def log_interp1d(xx, yy, kind='linear'):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind, fill_value="extrapolate")
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

def get_all_egb_data(energies, deltae):
    energy_range = np.array([(.1, .14), (.14, .2), (.2, .28), (.28, .4), (.4, .57), (.57, .8), (.8, 1.1), (1.1, 1.6), (1.6, 2.3), (2.3, 3.2), (3.2, 4.5), (4.5, 6.4), (6.4, 9.1), (9.1, 13), (13, 18), (18, 26), (26, 36), (36, 51), (51, 72), (72, 100), (100, 140), (140, 200), (200, 290), (290, 410), (410, 580), (580, 820)])*1e3 #GeV to MeV
    egb_intensity = np.array([3.7e-6, 2.3e-6, 1.5e-6, 9.7e-7, 6.7e-7, 4.9e-7, 3e-7, 1.8e-7, 1.1e-7, 6.9e-8, 4.2e-8, 2.6e-8, 1.7e-8, 1.2e-8, 6.8e-9, 4.4e-9, 2.7e-9, 1.8e-9, 1.1e-9, 6.2e-10, 3.1e-10, 1.9e-10, 8.9e-11, 6.3e-11, 2.1e-11, 9.7e-12])
    middle_bin = []
    bin_width = []
    for i in range(0, len(energy_range)):
        low_e = np.log10(energy_range[i][0])
        high_e = np.log10(energy_range[i][1])
        difference = np.abs((low_e+high_e)/2)
        middle_bin.append(10**(difference))
        bin_width.append(np.abs(energy_range[i][1]-(10**(difference)))) 
    return middle_bin, bin_width, egb_intensity

def get_all_egb(energies, deltae):
    energy_range = np.array([(.1, .14), (.14, .2), (.2, .28), (.28, .4), (.4, .57), (.57, .8), (.8, 1.1), (1.1, 1.6), (1.6, 2.3), (2.3, 3.2), (3.2, 4.5), (4.5, 6.4), (6.4, 9.1), (9.1, 13), (13, 18), (18, 26), (26, 36), (36, 51), (51, 72), (72, 100), (100, 140), (140, 200), (200, 290), (290, 410), (410, 580), (580, 820)])*1e3 #GeV to MeV
    egb_intensity = np.array([3.7e-6, 2.3e-6, 1.5e-6, 9.7e-7, 6.7e-7, 4.9e-7, 3e-7, 1.8e-7, 1.1e-7, 6.9e-8, 4.2e-8, 2.6e-8, 1.7e-8, 1.2e-8, 6.8e-9, 4.4e-9, 2.7e-9, 1.8e-9, 1.1e-9, 6.2e-10, 3.1e-10, 1.9e-10, 8.9e-11, 6.3e-11, 2.1e-11, 9.7e-12])
    middle_bin = []
    bin_width = []
    for i in range(0, len(energy_range)):
        low_e = np.log10(energy_range[i][0])
        high_e = np.log10(energy_range[i][1])
        difference = np.abs((low_e+high_e)/2)
        middle_bin.append(10**(difference))
        bin_width.append(np.abs(energy_range[i][1]-(10**(difference)))) 
        

    log_interp = log_interp1d(middle_bin, egb_intensity/bin_width, kind='linear')
    
    '''
    print(egb_intensity[2]/bin_width[2]*deltae[6])
    print(energies[6])
    x_trapz = np.logspace(np.log10(np.nanmin(energies)), np.log10(np.nanmax(energies)), num = 100)
    plt.scatter(middle_bin, egb_intensity/bin_width)
    plt.plot(x_trapz, log_interp(x_trapz), color = 'red')
    plt.scatter(energies[6], log_interp(energies[6]), color = 'green')
    plt.yscale('log')
    plt.xscale('log')
    '''
    
    counts = []
    #only want energies from 3 onward (about lowest at 80 MeV)
    for x in range(0, len(energies)):
        highest_val = energies[x]+deltae[x]
        lowest_val = energies[x]-deltae[x]
        x_trapz = np.logspace(np.log10(lowest_val), np.log10(highest_val), num = 600)
        #counts.append(log_interp(energies[x])*deltae[x])
        total_counts = np.trapz(log_interp(x_trapz), x = x_trapz)
        
        counts.append(total_counts)  
    return counts #returns counts per cm^2 per sec per str

import numpy as np
from astropy.io import fits

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#get all ktests

def get_ktests(save= False, dmfilename = 'yield_DS_keith40.dat', massdm = 40, filenam = 'ktestsfile1MeVfin5yrs4.fits'):

    ktest_array = []
    importlib.reload(aaa)

    energies = np.copy(central_energies)
    #cross_sec normalized to 1.4e-26
    deltae_cut = np.copy(deltae)

    egb_counts = get_all_egb(energies, deltae)/deltae_cut
    counting = 0

    energiesforBH = np.logspace(np.log10(.05), np.log10(1e7), num = 1000) #in MeV
    #eventually make it so this is only calculated once per mass_bh
    #interpolate the luminosity

    print(len(energies))
    for energyidx in range(0, len(energies)):
        print('---')
        print(energyidx)
        print(energies[energyidx])
        print('---')
        pitest = poisson_dist(2, int(energyidx))
        icstest = poisson_dist(4, int(energyidx))
        bremtest = poisson_dist(0, int(energyidx))
        #EGB counts, we have at each energy bin in units of per cm^2 per s per str per MeV
        egbtest = poisson_dist(np.nan, int(energyidx), egb = True, counts = egb_counts[energyidx])


        #Point Sources
        pointstest = poisson_dist(np.nan, energyidx, points = True) 


        #Dark matter
        importlib.reload(dmj) 
        darkmtest = poisson_dist(np.nan, int(energyidx), dm = True, analyze_data = False, filenm = dmfilename, dm_mass = massdm)
        darkmtest[np.isnan(darkmtest)] = 0
        
        energy_here = central_energies[energyidx]
        acceptance_for_poisson = acceptance_interp(energy_here)
        #ktest1 = pitest+icstest+bremtest+darkmtest+egbtest+pointstest+darkmtest
        print(np.sum(pitest))
        print(np.sum(icstest))
        print(np.sum(bremtest))
        print(np.sum(egbtest))
        print(np.sum(pointstest))
        print(np.sum(darkmtest))
        ktest1 = simulated_data(int(energyidx), list([pitest, icstest, bremtest, egbtest, pointstest, darkmtest]))
        ktest_array.append(ktest1)

        counting += 1
        
    if save:
        hdu = fits.PrimaryHDU(ktest_array)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filenam)
    return ktest_array

def get_loglikeli(MYDIR, energyidx, MYDIR1, templates):
    print(MYDIR)
    print(energyidx)
    print(MYDIR1)

    #subprocess.run(["python3", "./pymultinest_chains/multinest_marginals.py", "./{MYDIR1}/chain{energyidx}/{energyidx}"])
    #Open the file
    #path_to_this_file = MYDIR + '/' + str(energyidx) + 'stats.json'
    path_to_this_file = MYDIR + '/' + str(energyidx) + 'stats.dat'

    #f = open(path_to_this_file)
    #filehere = json.load(f)
    f = np.loadtxt(path_to_this_file, skiprows = 4, max_rows = 4).T[1]
    testpoints0 = (10**f[0]) #pitest
    testpoints1 = (10**f[1]) #icstest
    testpoints2 = (10**f[2]) #bremtest
    testpoints3 = (10**f[3]) #points
    
          
    constants=[testpoints0, testpoints1, testpoints2, testpoints3]
    #print('constants: ')
    #print(constants)
    return constants
    #constants, pitest, icstest, bremtest, egbtest, pointstest, ktest, darkmtest, blackholetest
    #likehere = likelihood_poisson_forsigmav(constants, templates[0], templates[1], templates[2], templates[3], templates[4], templates[5], templates[6], templates[7])
    #return likehere

def likelihood_poisson_forsigmav(constants, piflux, icsflux, bremflux, egbtest, pointstest, ktesthere, darkmtest, blackholetest):
    #Constants are my consants for pi, ics, brem, and point sources. the constant in front of egb is 1.
    #The 'constant' in front of the dark matter is already incorporated by changing sigmav
    #This is my value of lambda
    lambd = constants[0]*piflux+constants[1]*icsflux+constants[2]*bremflux+constants[3]*pointstest+egbtest+darkmtest+blackholetest

    
    #my value of k (ktest) is the poisson drawn values from sum of all the templates except dm
    fprob = -scipy.special.gammaln(ktesthere+1)+ktesthere*np.log(lambd)-lambd #log likelihood of poisson
    #scipy.special.gammaln is for the log of a factorial
    return -2*np.nansum(fprob) #the sum of all the log likelihoods for each spatial point. 

#find index in energy list closest to this energy
def find_nearest(energiesforBH, central_energies, lum):
    array = np.array(energiesforBH[::-1])
    backwardslum = np.array(lum[::-1])
    plt.plot(energiesforBH, lum*np.array(energiesforBH)**2)
    idxs = len(energiesforBH)-np.where(backwardslum*np.array(array)**2 >= np.nanmax(backwardslum)*np.array(array)**2/3e2)[0]
    value = energiesforBH[idxs[0]]
    
    array = np.asarray(central_energies)
    idx = (np.abs(array - value)).argmin()
    
    if idx < 2:
        return 2
    else:
        return idx



def get_likelihoodat(fbh, ktestname, name_of_file = 'testingnew/', dmfilename = 'yield_DS_keith40.dat', massdm = 40, blackholem = 2e16, gam = 1.6):
    ktest_array = readfile(ktestname)[0].data
    energies = np.copy(central_energies)
    deltae_cut = np.copy(deltae)

    test_cross = 1.4e-26


    lum = (photon_spectrum.get_integral(egamma_values, mass_bh = blackholem)[1]) #units of photons per MeV per sec per BH
    lum_interp = log_interp1d(energiesforBH, lum, kind='linear') #integrate dn/dE in units of MeV

    #first find likelihood where fbh = 0

    egb_counts = get_all_egb(energies, deltae)/deltae_cut #units of counts per cm^2 per sec per str per MeV
    ##These are the names of the parameters we are fitting.
    parameters = ['a0', 'a1', 'a2', 'a3']
    folder = './pymultinest_chains/' + str(name_of_file)
    livepoints = 400

    '''Check if directory exists, if not, create it'''
    import os

    # You should change 'test' to your preferred folder.
    MYDIR1 = (folder + "fbh" + str(fbh))
    folder_name = "fbh" + str(fbh)
    CHECK_FOLDER = os.path.isdir(MYDIR1)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(MYDIR1)
        print("created folder : ", MYDIR1)

    else:
        print(MYDIR1, "folder already exists.")


    likefbh = 0
    print('-----------')
    print('gamma for BHs: {}'.format(gam))
    print('mass for BHs: {}'.format(blackholem))
    highest_range = find_nearest(energiesforBH, central_energies, lum)
    print('highest energy going to:')
    print(central_energies[highest_range])
    print('index:')
    print(highest_range)
    print('-------------------------')
    for energyidx in range(0, highest_range):
        print(energyidx)
        print('energy here: {}'.format(energies[energyidx]))
        # You should change 'test' to your preferred folder.
        MYDIR = (MYDIR1 + "/chain" + str(energyidx))
        CHECK_FOLDER = os.path.isdir(MYDIR)

        # If folder doesn't exist, then create it.
        if not CHECK_FOLDER:
            os.makedirs(MYDIR)
            print("created folder : ", MYDIR)
        else:
            print(MYDIR, "folder already exists.")
        print(energyidx, energies[energyidx])

        #photons per pixel
        pitest = poisson_dist(2, int(energyidx))
        icstest = poisson_dist(4, int(energyidx))
        bremtest = poisson_dist(0, int(energyidx))

        #Point Sources
        pointstest = poisson_dist(np.nan, energyidx, points = True)

        #EGB counts, we have at each energy bin in units of per cm^2 per s per str per MeV
        egbtest = poisson_dist(np.nan, int(energyidx), egb = True, counts = egb_counts[energyidx], cross_section = test_cross)

        #Dark matter
        darkmtest = poisson_dist(np.nan, int(energyidx), dm = True, cross_section = test_cross, analyze_data = False, filenm = dmfilename, dm_mass = massdm)
        darkmtest[np.isnan(darkmtest)] = 0

        #Black Holes
        blackholetest = poisson_dist(np.nan, int(energyidx), cross_section = test_cross, dm_bh = True, evapbh = True, blackholem = 2e16, luminterpolated = lum_interp, gamm = gam)*fbh
        blackholetest[np.isnan(blackholetest)] = 0

        #original data
        ktest = ktest_array[energyidx] #ktest does not include black holes
        
        args = (pitest, icstest, bremtest, pointstest, egbtest, darkmtest, blackholetest, ktest)
        #print('-----------------------')
        #print(np.nansum(ktest))
        #print(np.nansum(pitest+ icstest+ bremtest+ pointstest+ egbtest+ darkmtest))
        #print(np.nansum(pitest+ icstest+ bremtest+ pointstest+ egbtest+ darkmtest+blackholetest))
        #print('-----------------------')
        
        cube_limits = [(-6, 12), (-4, 8), (-4, 8), (-4, 8)]
        
        def prior(cube, ndim, nparams):
            #cube[0] = (cube[0]*np.abs(np.log10(1.1)-np.log10(.9)) - np.log10(.9)) #from 1e-4 to 1e6 apparently
            cube[0] = (cube[0]*cube_limits[0][1] + cube_limits[0][0])
            cube[1] = (cube[1]*cube_limits[1][1] + cube_limits[1][0])
            cube[2] = (cube[2]*cube_limits[2][1] + cube_limits[2][0])
            cube[3] = (cube[3]*cube_limits[3][1] + cube_limits[3][0])
            
    
        ##This is the loglikelihood function for multinest – it sends the cube,
        #and then a bunch of different arrays that were used in fitting, but were constant, to the pymultinest code
        def loglikelihood_formulti(cube, ndim, nparms):
            return likelihood_poisson_multinest(cube, ndim, nparms)
        
        #cube, pitest, icstest, bremtest, egbtest, pointstest, darkmtest, blackholetest, ktest
        def likelihood_poisson_multinest(cube, ndim, nparams):
            pi, ics, brem, points, egb, darkm, blackhole, k = args
            #print(np.sum(pi+ics+brem+points+egb+darkm+blackhole))
            
            a0 = 10**cube[0]
            a1 = 10**cube[1]
            a2 = 10**cube[2]
            a3 = 10**cube[3]
            
            #print(a0, a1, a2, a3)
    
            lamb = a0*pi+a1*ics+a2*brem+a3*points+egb+darkm+blackhole #egb is constant
            fprob = -scipy.special.gammaln(k+1)+k*np.log(lamb)-lamb #log likelihood of poisso
            #print('---------------------------------')
            return 2*np.nansum(fprob) #perhaps add negative back, perhaps add 2 back?
        
        def likelihood_poisson(a0log, a1log, a2log, a3log):
            pi, ics, brem, points, egb, darkm, blackhole, k = args
            #print(np.sum(pi+ics+brem+points+egb+darkm+blackhole))
            #print(a0, a1, a2, a3)
            a0 = 10**a0log
            a1 = 10**a1log
            a2 = 10**a2log
            a3 = 10**a3log
    
            lamb = a0*pi+a1*ics+a2*brem+a3*points+egb+darkm+blackhole #egb is constant
            fprob = -scipy.special.gammaln(k+1)+k*np.log(lamb)-lamb #log likelihood of poisso
            #print('---------------------------------')
            return -2*np.nansum(fprob) #perhaps add negative back, perhaps add 2 back?

        finals = pymultinest.run(loglikelihood_formulti, prior, int(len(parameters)), outputfiles_basename=MYDIR+"/"+ str(energyidx), n_live_points=livepoints, resume=True, verbose=True)
        json.dump(parameters, open(MYDIR+'/' + str(energyidx) + 'params' +'.json', 'w'))
        const = get_loglikeli(MYDIR, energyidx, MYDIR1, [pitest, icstest, bremtest, egbtest, pointstest, ktest, darkmtest, blackholetest])
        m = Minuit(likelihood_poisson, a0log=np.log10(const[0]), a1log = np.log10(const[1]), a2log = np.log10(const[2]), a3log = np.log10(const[3]))
        lowval = 1
        highval = 1

        m.limits = [(np.log10(const[0])-lowval, np.log10(const[0])+highval), (np.log10(const[1])-lowval, np.log10(const[1])+highval), (np.log10(const[2])-lowval, np.log10(const[2])+highval), (np.log10(const[3])-lowval, np.log10(const[3])+highval)]
        m.migrad()
        m.hesse()
        
        print(m.values)
        
        fin_consts = [10**m.values[0], 10**m.values[1], 10**m.values[2], 10**m.values[3]]
        
        print('constants from minuit: ')
        print(fin_consts)
        likelihood = likelihood_poisson(m.values[0], m.values[1], m.values[2], m.values[3])
        #print('current likelihood: {}'.format(likelihood))
        likefbh += likelihood
    print('final likelihood: {}'.format(likefbh))
    return likefbh

def twosig_fbh(testfbh, likefbh0, ktest, name_of_file = 'testingnew/', dmfilename = 'yield_DS_keith40.dat', massdm = 40, blackholem = 2e16, gam = 1.6):
    print(10**testfbh)
    return root_find(float(10**testfbh), ktest, namefile = name_of_file, dmfname = dmfilename, mdm = massdm, bhm = blackholem, gamma = gam)-4-likefbh0

def root_find(fbhtesting, k, namefile = 'testingnew/', dmfname = 'yield_DS_keith40.dat', mdm = 40, bhm = 2e16, gamma = 1.6):
    
    likefbh = get_likelihoodat(fbhtesting, k, name_of_file = namefile, dmfilename = dmfname, massdm = mdm, blackholem = bhm, gam = gamma)
    print('likelihood here: {}'.format(likefbh))
    return np.abs(likefbh)

    




    



filelist1 = ['Bremss_00320087_E_50-814008_MeV_healpix_128.fits', 'Bremss_SL_Z6_R20_T100000_C5_E_50-814008_MeV_healpix_128.fits', 'pi0_Model_A_E_50-814008_MeV_healpix_128.fits', 'pi0_Model_F_E_50-814008_MeV_healpix_128.fits', 'ICS_Model_A_E_50-814008_MeV_healpix_128.fits', 'ICS_Model_F_E_50-814008_MeV_healpix_128.fits']
filelist = ['bremss_healpix_reshuffled_53templates_0511MeV_reran.fits', 'bremss_healpix_reshuffled_53templates_0511MeV_reran.fits', 'pi0_decay_healpix_reshuffled_53templates_0511MeV_reran.fits', 'pi0_decay_healpix_reshuffled_53templates_0511MeV_reran.fits', 'ics_isotropic_healpix_reshuffled_53templates_0511MeV_reran.fits', 'ics_isotropic_healpix_reshuffled_53templates_0511MeV_reran.fits']

bremss_reopen = fits.open('ics_isotropic_healpix_reshuffled_53templates_0511MeV_reran.fits')
bremssenergy_list = np.array([bremss_reopen[i].header['MID_E'] for i in range(len(bremss_reopen) - 1)])
central_energies = np.copy(bremssenergy_list)


point_sources = ['PS_nonfloat_511MeV_0-52.fits']

#for astrogam, use 5 years
exposure_time = 1.578e8

acceptance_interp = aaa.get_acceptance_interp() #put in the energy in MeV!
angle_interp = aaa.get_angle_interp()



#Egammas for evaporating black holes
energiesforBH = np.logspace(np.log10(.05), np.log10(1e7), num = 1000)
#egamma_values = photon_spectrum.get_egammas(energiesforBH)
egamma_values = fits.open('egammavals.fits')[0].data

counts, deltae = get_darksusy_counts()

acceptances = []
cut_energy = np.copy(central_energies)
for energyidx in range(0, len(cut_energy)):

    energy_here =central_energies[energyidx]
    acceptance_forpoisson = acceptance_interp(energy_here) #in m^2*str, convert to cm^2
    acceptances.append(acceptance_forpoisson)
    
#ktest_array = readfile('ktests511.fits')[0].data
    


