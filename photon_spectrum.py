import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import scipy
from scipy.special import factorial as factorial
from scipy.special import factorial2 as doublefactorial
from scipy import interpolate
import scipy.integrate as integrate
from scipy.integrate import quad
from time import process_time

universal_energy = np.logspace(np.log10(.1), np.log10(1e6), num = 10000)
new_universal_energy = np.logspace(np.log10(.1), np.log10(1e6), num = 1000)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def dEdx(E, n_H):
    gamma = E/.511 #MeV/MeV
    beta = (np.sqrt(1-(1/gamma**2)))
    return (7.6e-26/beta**2)*(n_H/0.1)*(np.log(gamma)+6.6) #in MeV/cm

def sigma(E, n_H):
    r_e = 2.82e-13 #cm
    gamma = E/.511 #MeV/MeV
    beta = (np.sqrt(1-(1/gamma**2)))
    
    sig = (np.pi*r_e**2)/(gamma + 1)*((gamma**2+4*gamma+1)/(gamma**2-1)*np.log(gamma + np.sqrt(gamma**2-1))-(gamma + 3)/(np.sqrt(gamma**2-1)))
    return sig

def prob(E0, E, n_H):
    #E0 = range of E positron
    #E = range of positron ray (one number)
    
    #E = np.linspace(E, E0, num = 300)
    
    sig = sigma(E0, n_H)
    
    func = np.nan_to_num(sig/dEdx(E0, n_H))
    
    #x = np.asarray([np.trapz(func[:i], E0[:i]) for i in range(len(func))])
    x = np.empty(func.shape[0])
    y = np.empty(func.shape[0])
    x[0] = 0
    x[1:] = 0.5 * (E0[1:] - E0[:len(func)-1]) * (func[1:] + func[:len(func)-1])
    y = np.cumsum(x)
    return np.exp(-n_H*y)
    
def d_sigma(Egamma, Epositron):
    #Epositron = np.asarray(Epositron)
    r_e = 2.82e-13 #cm
    
    dsigma_arr = []
    
    #for i in Egamma:
    k = Egamma/.511 #in MeV/MeV, from the gamma rays
    gamma = Epositron/.511 #in MeV/MeV, from positron
    beta = (np.sqrt(1-(1/gamma**2))) #from the positron
    
    lessk = np.less(k,((gamma*(1-beta)+1)/2))
    greaterk = np.greater(k, ((gamma*(1+beta)+1)/2))
    lor = np.logical_or(lessk, greaterk)
    dsigma_prefactor = (np.pi*r_e**2)/(gamma**2*beta**2)
    dsigma = ((-(3+gamma)/(1+gamma)+(3+gamma)/k-1/(k**2))/(1-k/(1+gamma))**2-2)
    dsigmafin = [0 if lor[i] else dsigma_prefactor[i]*dsigma[i] for i in range(len(lor))]
    return np.asarray(dsigmafin)

#Now we have our working example for dsigma. Let's try one more time to get our integral working.

def diffuse_flux(E):
    return E*0.013*(E)**(-1.8) # cm−2 s −1 sr−1 MeV−1


def dphidk(n_H, Epos):
    phi = 0.000287#*(6.48e44)#photons cm−2 s−1 #using the fact that we get 10^50 positrons per year, find what flux must be
    #phi = get_IA_spectra()
    #phi = 1 #photons per black hole per second
    #converted to cm^2
    #phi511 = 1e-3 #photons cm−2 s−1
    f = 0.967 # ± 0.022

    
    E_gam = np.logspace(np.log10(.511), np.log10(Epos + 1), num = 20)
    minrange_Epos = .511
    #E_gam = np.logspace(np.log10(.2), np.log10(Epos + 1), num = 20)
    #minrange_Epos = .2
    range_Epos = np.logspace(np.log10(minrange_Epos), np.log10(Epos), num = 20000)
    
    total_val = []
    #temp_tot_prob = []
    #temp_tot_dedx = []
    temp_tot_dsig = []

    
    #E_gam and Epos are different. We integrate over Epos, but E_gamma is also relevant
    for gamma_energy in E_gam:
        temp_func = []
        
        dsig = d_sigma(gamma_energy, range_Epos)
        probability = (prob(range_Epos, gamma_energy, n_H))
        dE_dx = dEdx(range_Epos, n_H)
        temp_func.append(np.nan_to_num(dsig*probability/dE_dx))
        #temp_tot_dedx.append(np.trapz(1/dE_dx, x = range_Epos))
        #temp_tot_prob.append(np.trapz(probability, x = range_Epos))
        temp_tot_dsig.append(np.trapz(dsig, x = range_Epos))
        total_val.append(np.trapz(temp_func, x = range_Epos))
    
    #plt.figure(figsize = (9, 10))
    #plt.plot(E_gam, temp_tot_dsig)
    #plt.ylim(1e-26, 1e-24)
    #plt.xscale('log')
    total_val = np.asarray(total_val)
    return np.transpose(phi/(1-3/4*f)*n_H/1/2*total_val/.511)[0], E_gam

def get_EdphidE(n_H, Epos):
    dphidE1, energy = dphidk(n_H, Epos) #3 MeV
    #dphidE3 = dphidk(0.1, 3)
    #dphidE10 = dphidk(0.1, 10)
    
    y_en = diffuse_flux(energy)#*1e4
    #dphidE = (dphidE1*energy+y_en)*1e4
    #dphidE = (dphidE1*energy)#*1e4
    dphidE = dphidE1#/energy #units of photons cm^-2 s^-1 MeV^-1

    
    #we want it to be zero where the baseline goes flat
    
    #interpolate it first for points at and after .511, then add back the array before .511 down to .2 as 0
    dphidE = interpolate.interp1d(energy, dphidE, kind = 'cubic', bounds_error = False, fill_value = 0)
    
    snipped_new_universal_energy = np.where(new_universal_energy >= .511)[0][0]
    energy_before511 = new_universal_energy[:snipped_new_universal_energy]
    energy_after511 = new_universal_energy[snipped_new_universal_energy:]
    
    
    my_func = dphidE(energy_after511)
    
    
    try:
        idx = np.argwhere(my_func <=1e-92)[0][0]
        #print(idx)
        #print(universal_energy[idx])
        my_func[idx:] = 0
    except IndexError:
        #print('ope')
        pass
    
    my_func = np.asarray(list(np.zeros(len(energy_before511)))+list(my_func))
    
    
    
    idx2 = np.argwhere(my_func < 0)
    my_func[idx2] = 0
    
    #my_func = my_func*(1.735e43)**(-1) #multiply by s/positron
    
    return my_func, new_universal_energy #returns in units of  photons MeV^-1  per positron
    #apparently is in units of photons per positron per energy

def inverseT(mass):
    return mass*1e-14/0.106 #GeV

def temptocm2(T):
    mass = 0.106/(1e-14*T) #grams
    #convert to planck mass
    planckmass = (mass/2.18e-5)**2*(1.62e-33)**2 #in cm^2
    return planckmass

def gtoGeV(mass):
    energy = 0.106/(1e-14*mass)
    return energy

def GeVtog(temp):
    if temp == 0:
        return 0
    return 0.106/(1e-14*temp)



def high_sigma(T, E, spin):
    return 27*np.pi*(temptocm2(T))

def low_sigma(T, E, spin):
    if (spin == 0):
        sig = sigma0(spin, T)
    if (spin == 1/2):
        sig = sigmahalf(spin, T, E)
    if (spin == 1):
        sig = sigma1(spin, T, E)
    if (spin == 2):
        sig = sigma2(spin, E, T)
    
    #grab the sigma based on the spin of the particle
    #print(sig[0:10])
    return sig

def sigma0(spin, T):
    #dan page paper
    A = area(T)
    sigma = A
    return sigma
    
def sigmahalf(spin, T, E):
    #electron case
    return 2 * np.pi*(temptocm2(T))
    
def sigma1(spin, T, E):
    #photon case, in dan's paper
    sig = (12/9)*area(T)*temptocm2(T)*(E)**2 * (1/1.97e-14)**2
    return sig
    
def sigma2(spin, E, T):
    sig = (16/225) * area(T) *5*(temptocm2(T)**2)*(E)**4 * (1/1.97e-14)**4# * (1/1.22e19)**2
    return sig

def area(T):
    return 16*np.pi * temptocm2(T)

def plusminus(spin):
    if (spin == 1/2):
        return 1
    if ((spin == 0) | (spin == 1) | (spin == 2)):
        return -1
    
def get_function(T, E, spin, particle_type, library):
    equation = []
    has_mass = library[particle_type][4]
    particle_mass = library[particle_type][0]
    '''
    try:
        length_T = len(T)
    except TypeError:
        T = [T]
    '''
    for x in range(0, 1):
        highlim = 27*np.pi*(temptocm2(T))
        lowlim = low_sigma(T, E, spin)
        sig = interp_func(highlim, lowlim, E, T, spin)
        #print(interp_func(highlim, lowlim, .32, T[x], spin)*0.32**3/(2*np.pi**2*(np.exp(0.32/T[x])+1)))
        conditional1 = plusminus(spin)
        if not has_mass:
            equation.append((1/2/np.pi**2)*E**2*sig/(np.exp(E/T)+conditional1))
        else:
            equation.append((1/2/np.pi**2)*E*(E**2-particle_mass**2)**(1/2)*sig/(np.exp(E/T)+conditional1))

            equationedit = equation[0]
            #print(equationedit)
            equationedit = np.where(np.isnan(equationedit), 0, equation)
            equation = np.copy(equationedit)
            #print(equation)
        
    return np.asarray(equation)

def interp_func(highlim, lowlim, E, T, spin):
    if (spin == 0):
        return (highlim-(highlim-lowlim)*np.exp(-1.0*(E/(5*T))**2.0))
    if (spin == 1/2):
        return (highlim-(highlim-lowlim)*np.exp(-1.0*(E/(5*T))**2.0))
    if (spin == 1):
        return (highlim-(highlim-lowlim)*np.exp(-1.0*(E/(6*T))**4.0))
        #return (highlim-(highlim-lowlim)*np.exp(-1.0*(E/(5*T))**2.0))
    if (spin == 2):
        return (highlim-(highlim-lowlim)*np.exp(-1.0*(E/(10*T))**8.0))
    
def unit_factor(particle, spin, library, inverse_sec = False):

    planckconst = 1/(6.582e-25) ##s**-1 GeV**-1
    conversionconst = (1/(1.97e-14))**2#*1e6 ##GeV**-2 cm **-2 converted to MeV^-2
    ergs100mev = 6242
        
    carrfactor = get_typing(particle, spin, library)
    
    if inverse_sec:
        factor = planckconst*conversionconst*carrfactor
    else:
        #use this one if you want to convert to Carr paper
        factor = planckconst*conversionconst/ergs100mev*carrfactor

    #print('my factor: ' + str(factor))
    
    return factor

def get_typing(particle, spin, library):
    weird_factor = 1
    degreesOF = library[particle][2]
    if spin == 1/2:
        weird_factor = (7/8)
    return weird_factor * degreesOF
    

#let's first just create an interpolation between the low energy and high energy of a particle.
def get_high_and_low(energy_list, mass, particle, library):
    temp = (1/inverseT(mass))
    #print('temperature: {}'.format(temp))
    spin = library[particle][1]
    particle_mass = library[particle][0]
    
    heavy_particles = ['electron', 'muon', 'tau', 'top', 'bottom', 'up', 'down', 'strange', 'charm', 'zboson', 'wboson', 'higgs']
    
    #if particle in heavy_particles:
        #energy_list = np.sqrt(energy_list**2-library[particle][0]**2)
    
    combined = get_function(temp, energy_list, spin, particle, library)
    combined = np.nan_to_num(combined)[0]
    
    #plt.plot(energy_list, combined*energy_list)
    #looking good here, still in GeV

    
    #also need to convert this to MeV not GeV
    
    #interp_energy = interpolate.interp1d(energy_list, combined*energy_list, kind = 'cubic')
    #print(interp_energy)
    #plt.plot(energy_list, interp_energy(energy_list))
    #asdfasd
    
    
    #now we modify our arrays to make them dn/de
    factor = unit_factor(particle, spin, library, inverse_sec = True)
    

    #interp_en = factor * interp_energy(energy_list)[0]
    
    #print(interp_en)
    #this gives us E*dNDe
    #integrate this to get total power
    #interp_en = factor*combined[0]*energy_list
    #interp = interpolate.interp1d(energy_list, combined*energy_list, kind = 'cubic')
    #interp_en = factor * combined*energy_list
    
    #switch back to this when not doing the integral. remember it returns in /GeV, so need to convert it to /MeV
    interp_en = factor * combined/1e3# * energy_list #need to convert to /MeV #this is now in /s/MeV
    #interp_en = factor * combined * energy_list #need to use 1e3 since energy list is still in GeV
    
    #plt.plot(energy_list*1e3, interp_en)
    
    
    return interp_en#, np.nan_to_num(interp), factor


#particle dictionary goes name, mass, spin, N, charge,  has mass, charge states,  mass is in GeV
particle_dict_aboveQCD = {
    "muon": [105.6583745*1e-3, 1/2, 4, 1/2, True, 2, 1.4547263377121311], #started with 4
    "up": [2.3*1e-3, 1/2, 12, 1/2, True, 2, 1.4547263377121311], #for all quarks!! Started at 12, maybe should be 12*6 quarks? 
    "down": [4.8*1e-3, 1/2, 12, 1/2, True, 2, 1.4547263377121311],
    "strange": [95*1e-3, 1/2, 12, 1/2, True, 2, 1.4547263377121311],
    "charm": [1275*1e-3, 1/2, 12, 1/2, True, 2, 1.4547263377121311],
    "top":[173210*1e-3, 1/2, 12, 1/2, True, 2, 1.4547263377121311],
    "bottom":[4180*1e-3, 1/2, 12, 1/2, True, 2, 1.4547263377121311],
    "gluon": [1.3*1e-3, 1, 16, 1, True, 8, 0.8970602892949766], #started with 16
    "tau": [1776.86*1e-3, 1/2, 4, 1/2, True, 2, 1.4547263377121311], #started with 4
    "higgs": [125, 0, 1, 0, True, 1, 1.3298399353802985], 
    "wboson":[80.4, 1, 4, 1, True, 2, 0.8970602892949766], #said 6, Dan says 4 ¯\_(ツ)_/¯
    "zboson":[91.26, 1, 3, 0, True, 1, 0.8970602892949768],
    #"pioncharged": [140*1-3, 0, 2, 1, True],
    #"pion0": [135*1e-3, 0, 1, 0, True],
    #"eta":[547.853*1e-3, 0, 1, 0, True],
    #"etaprime":[957.66*1e-3, 0, 1, 0, True ],
    #"rhocharged":[775.4*1e-3, 1, 6, 0, True ],
    #"rho0":[775.49*1e-3, 1, 3, 0, True ],
    #"omega":[782.65*1e-3, 1, 3, 0, True],
    #"phi":[1019.445*1e-3, 1, 3, 0, True],
    "electron":[0.5109989461*1e-3, 1/2, 2, 1/2, True, 2, 1.4547263377121311], #started with 4, should be 2 for just electron or just positron
    "neutrino":[0.120e-9, 1/2, 6, 1/2, True, 1, 1.4547263377121311], #changed the neutrino one to 4, should be 6 when we aren't using page
    "graviton":[0, 2, 2, 0, False, 1, 1.1428714478337467],
    "photon":[0, 1, 2, 1, False, 1, 0.8970602892949766]
}

#particle dictionary goes name, mass, spin, N, charge,  has mass, charge states,  mass is in GeV
particle_dict_belowQCD = {
    "muon": [105.6583745*1e-3, 1/2, 4, 1/2, True, 2, 1.4547263377121311], #started with 4
    #"up": [2.3*1e-3, 1/2, 12, 1/2, True, 2, 1.4547263377121311], #for all quarks!! Started at 12, maybe should be 12*6 quarks? 
    #"down": [4.8*1e-3, 1/2, 12, 1/2, True, 2, 1.4547263377121311],
    #"strange": [95*1e-3, 1/2, 12, 1/2, True, 2, 1.4547263377121311],
    #"charm": [1275*1e-3, 1/2, 12, 1/2, True, 2, 1.4547263377121311],
    #"top":[173210*1e-3, 1/2, 12, 1/2, True, 2, 1.4547263377121311],
    #"bottom":[4180*1e-3, 1/2, 12, 1/2, True, 2, 1.4547263377121311],
    #"gluon": [1.3*1e-3, 1, 16, 1, True, 8, 0.8970602892949766], #started with 16
    "tau": [1776.86*1e-3, 1/2, 4, 1/2, True, 2, 1.4547263377121311], #started with 4
    "higgs": [125, 0, 1, 0, True, 1, 1.3298399353802985], 
    "wboson":[80.4, 1, 4, 1, True, 2, 0.8970602892949766], #said 6, Dan says 4 ¯\_(ツ)_/¯
    "zboson":[91.26, 1, 3, 0, True, 1, 0.8970602892949768],
    "pionplus": [.139570, 0, 1, 1, True, -1, 1.3298399353802985],
    "pionminus": [.139570, 0, 1, 1, True, -1, 1.3298399353802985],
    "pion0": [.1349766, 0, 1, 0, True, -1, 1.3298399353802985],
    "kaon":[.496, 0, 4, 0, True, -1, 1.32983993545754], #4 DOF
    #"eta":[547.853*1e-3, 0, 1, 0, True],
    #"etaprime":[957.66*1e-3, 0, 1, 0, True ],
    #"rhocharged":[775.4*1e-3, 1, 6, 0, True ],
    #"rho0":[775.49*1e-3, 1, 3, 0, True ],
    #"omega":[782.65*1e-3, 1, 3, 0, True],
    #"phi":[1019.445*1e-3, 1, 3, 0, True],
    "electron":[0.5109989461*1e-3, 1/2, 2, 1/2, True, 2, 1.4547263377121311], #started with 4, should be 2 for just electron or just positron
    "neutrino":[0.120e-9, 1/2, 6, 1/2, True, 1, 1.4547263377121311], #changed the neutrino one to 4, should be 6 when we aren't using page
    "graviton":[0, 2, 2, 0, False, 1, 1.1428714478337467],
    "photon":[0, 1, 2, 1, False, 1, 0.8970602892949766]
}

def phase_transition(massBH):
    temp = (1/inverseT(massBH))
    
    if temp <= .3:
        return particle_dict_belowQCD
    if temp > .3:
        return particle_dict_aboveQCD

    
def get_high_and_low_normalized(energy_list, mass, particle, library):
    arrays = get_high_and_low(energy_list, mass, particle, library) #in per second per MeV
    
    #now we want to normalize it:

    my_integral = np.trapz(arrays, x = energy_list)
    dans_integral = get_normalization(mass, particle, library)
    #print("dan's integral in GeV/s: {}".format(dans_integral))
    #print('normalization: {}'.format(dans_integral/my_integral))
    normal = library[particle][6]
    #print('normal as saved: {}'.format(normal))
    
    
    return arrays*normal#/6242*energy_list*1e3 #6242 is for weird page units, positrons MeV^-1 s^-1 per BH 
    
def get_normalization(mass, particle, library):
    
    dof = library[particle][2]
    spin = library[particle][1]
    
    if spin == 0:
        gstar = dof * 1.82
    if spin == 1/2:
        gstar = dof * 1
    if spin == 1:
        gstar = dof * .41
    if spin == 2:
        gstar = dof * 0.05
        
    #print('gstar: {}'.format(gstar))
    
    return 8.2e6*(gstar/108)*(1e10/mass)**2*5.61e23 #to GeV/sec

#try again with the code:

def get_egammas(energy_list):
    n_h = 0.1
    egammas = []
    for i in range(0, len(energy_list)):
        #print(energy_list[i])
        egammas.append(get_EdphidE(0.1, energy_list[i])[0]) #photons MeV^-1  per positron

    return egammas


def get_integral(egammas, mass_bh = 5e14):
    
    #integral_energies = np.linspace(-29.16, 247.8, num = 100)
    #E_gam_energies = np.logspace(np.log10(.511), np.log10(300), num = 100)
    
    #new universal energy is new_universal_energy = np.logspace(np.log10(.511), np.log10(300), num = 100)
    E_gam_energies = np.copy(new_universal_energy)
    integral_energies_noti = np.copy(new_universal_energy)
    
    library = phase_transition(mass_bh)
    #first get carr curve for mass
    #returns in units of 1/s *1/MeV
    arrays = get_high_and_low_normalized(new_universal_energy*1e-3, mass_bh, 'electron', library) 
    flux_photon = get_high_and_low_normalized(new_universal_energy*1e-3, mass_bh, 'photon', library)
    norm_IA = float(get_IA_norm(arrays, mass_bh, new_universal_energy))
        
    #let's do the final FSR contribution: 
    fsr = get_FSR_spectra(E_gam_energies, arrays)

    dNgamma = []
    #import time
    #one = time.time()
    for y in range(0, len(E_gam_energies)):
        E_gam = E_gam_energies[y]
        current_value = 0
        #print('E_gam = {}'.format(E_gam))
        #print(egammas[y+1])
        for x in range(y, len(integral_energies_noti)):
            #units of MeV
            i = x+1
            #i = np.log10(E_e)*100
            #E_e = get_E(i)
            #print('positron energy: {}'.format(E_e))
            #print('positron list. : {}'.format(integral_energies_noti[x]))
            #delta_Ee = (10**((i+0.5)/100)-10**((i-0.5)/100)) #should be in MeV
            delta_Ee = get_deltaE(i)
            #print(delta_Ee)
            #if x == 43:
            dNdE_e = 2*arrays[x] #should be in positrons s^-1 MeV^-1 per black hole, need to mulitply by 2
            #else:
            #dNdE_e = 0

            dN_gammadE_gamma = egammas[x][y] #should be in photons per energy per electron
            current_value += (delta_Ee*dNdE_e*dN_gammadE_gamma)
        dNgamma.append(current_value)
    #two = time.time()
    #print(two-one)
        
    original_integral = np.trapz(np.asarray(dNgamma), x = E_gam_energies)
    normalization = norm_IA/original_integral

    
    #return E_gam_energies, np.asarray(dNgamma)*normalization, flux_photon, fsr
    #return E_gam_energies, flux_photon
    #return E_gam_energies, np.asarray(dNgamma)*normalization, flux_photon, fsr
    return E_gam_energies, np.asarray(dNgamma)*normalization + flux_photon + fsr
    #return E_gam_energies, dNgamma + flux_photon #flux photon is in photons per MeV per second per BH, 
    #dNgamma is in photons per MeV per second per BH
    #return E_gam_energies, dNgamma*np.asarray(num_of_BHs)/new_universal_energy, flux_photon
    
def get_FSR_spectra(e_gam, positron_arrays):
    
    fsr_final = []
    for i in range(0, len(e_gam)):
        dN_FSR = get_dN_FSR(e_gam[i], e_gam)
        dN_FSR_bad = (np.isnan(dN_FSR) | np.isinf(np.abs(dN_FSR)))
        dN_FSR[dN_FSR_bad] = 0
        integrate_this = positron_arrays*dN_FSR
        FSR = np.trapz(integrate_this, e_gam)
        fsr_final.append(FSR)
    return 2*np.asarray(fsr_final)
    
def get_dN_FSR(E_gamma, pos_energy):
    alpha = 1/137
    m_i = .511 #MeV
    E_f = np.copy(pos_energy) #MeV
    Q_f = 2*E_f #MeV
    x = 2*E_gamma/Q_f
    mu_i = m_i/Q_f
    
    P = (1+(1-x)**2)/x
    log = (np.log((1-x)/(mu_i**2))-1)
    
    return alpha/np.pi/Q_f*P*log
    

def get_IA_norm(positron_arrays, mass_bh, energy):
    probability = prob(energy, 0, 0.1)
    for_integration = np.asarray(positron_arrays)*(1-np.asarray(probability))

    integral = 2*np.trapz(for_integration, x = energy)
    return integral

def integral_EdNdE(E_energies, dNdE, r_energies = [.2, .6]):
    y = dNdE#*E_energies
    #print('whole thing: {}'.format(np.trapz(dNdE, x = E_energies)))
    
    good_values = np.where((E_energies <= float(r_energies[1])) & (E_energies >= float(r_energies[0])))
    
    new_E_energies = E_energies[good_values]
    new_dNdE = dNdE[good_values]
    
    return np.trapz(new_dNdE, x = new_E_energies)
        
def get_deltaE(i):
    return get_E(i+0.5)-get_E(i-0.5)
    
def get_E(i):
    #c = 1.072031549586548923679060665362035225048923679060 #for length of 100 of new_universal_energy
    #c = 1.0069166927592954990215264187 #length 1000
    #c = (2.01572525e-01/.2)#length 1000 starting at .2 MeV
    c = (0.02726689227030977/0.027) #length 1000 starting at 0.027 MeV (or 27 keV)
    return .027*c**(i-1)
    return .2*c**(i-1)
    return .511*c**(i-1)
        
#this is the function to return the integral from the cell above
def get_over_range(massBH, egamma_values, range_energies = [.2, .6]):

    energies_forslayter, dNdE_forslayter = get_integral(egamma_values, mass_bh = massBH)
    energy = np.trapz(dNdE_forslayter*energies_forslayter, x = energies_forslayter)
    #print(energy)
    integral_slayter = integral_EdNdE(energies_forslayter, dNdE_forslayter, r_energies = range_energies)
    
    #print('mass of black hole: {}'.format(massBH))
    
    return integral_slayter

def get_integral_over_range_electron(massBH, egamma_values, range_energies = [.511, 500]):
    library = phase_transition(massBH)
    arrays_electron = get_high_and_low_normalized(new_universal_energy*1e-3, massBH, 'electron', library)

    integral_slayter = integral_EdNdE(new_universal_energy, arrays_electron, r_energies = range_energies)
    return integral_slayter