# -*- coding: utf-8 -*-
"""
Created on Mon May  4 22:24:27 2020

@author: Lev Krayzman

This file contains functions for fitting the temperature dependence up to Tc
of superconducting resonators using Mattis-Bardeen theory.
It is based on Mathematica code of similar purpose by Gianluigi Catelani (and
possibly Matt Reagor).

The "material properties" section has some numbers that need to be modified for 
    different metals.
    
Set calc_sigmas = True at the beginning of "Compute complex conductivities using dtf"
    to recompute the tables of complex conductivities. This should only be necessary 
    if the tables that are supposed to come with this file have become lost,
    or they're discovered to be inadequate in some way. Otherwise, they are universal.
    
The actual fitting function defined in this code is BardeenMattisFit.
There are example usages in the main section.
    
Some references:
[1] Theory of the Anomalous Skin Effect in Normal and Superconducting Metals by
        Mattis, Bardeen -- the original paper
[2] Matt Reagor's thesis, section 5.1 -- describes this measurement for a 
        superconducting microwave cavity, brief overview
[3] J. Gao's thesis -- has an in-depth section on these calculations
[4] Superconducting Microresonators: Physics and Applications by Jonas Zmuidzinas -- 
        also has some descriptions, similar to J. Gao 
[5] Electromagnetic properties of superconductors, Exact Solution of the Mattis-Bardeen
        Equations for Bulk Material and Thin Films by Pöpel (1989) -- has both
        calculations and comparisons with a number of experiments for different
        metals in different regimes. Gao seems to follow this
[6] The Surface Impedance of Superconductors and Normal Conductors:
        The Mattis-Bardeen Theory by Turneaure, Halbritter, Schwettman --
        another description of the theories of superconductor impedance, including M&B
[7] Theory of Electromagnetic Properties of Strong-Coupling and Impure Superconductors II
        by S.B. Nam -- has expressions for sigmas as well as impedance as a function of them
        
Tinkham and Van Duzer also talk about this.
"""

###############################################################################
################################## IMPORTS ####################################
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
# Interpolation stuff
from scipy.interpolate import splrep, splev, RectBivariateSpline # interp2d
from lmfit import Parameters, report_fit
from lmfit import minimize as minimize
from scipy.optimize import fsolve
import pandas
import os
from sys import exit

###############################################################################
############################# Universtal constants ############################
###############################################################################
KB = 1.381e-23 # J/K
HBAR = 1.055e-34 # Js
MU0 = 4e-7 * np.pi # H/m = Ohm s/m

# Xs at T=0 is undefined due to divide by zero, instead compute it here:
ALMOST_ZERO = 0.06 # = T/Tc

# 1/tc is the 1.76 that is the ratio between Tc and gap
TC_OVER_GAP = np.exp(np.euler_gamma)/np.pi;
#TODO use experimental values from e.g. Van Duzer back cover, they differ up to 30%
# Nb: 3.8/2 = 1.9
# Ta: 3.6/2 = 1.8
# In: 3.6/2 = 1.8
# Al: 3.3/2 = 1.65

###############################################################################
############################# Material properties #############################
###############################################################################

# A note on Debye temperature: first, this is a material-dependent quantity.
# Second, it's also temperature, etc. dependent. Its value varies depending on
# where exactly you find it. Luckily, the following computation does not depend
# strongly on the exact value, as long as it's much larger than 2*Tc. Thus, 
# don't worry too much about finding exactly the right value. 

### WARNING ###
# I have included values for Nb which is not really a weakly-coupling
# superconductor. The fits will probably still mostly work, but be aware!
###         ###

t_Debye = 430; # K, bulk Al, book value.
# t_Debye = 109; # K, bulk In, book value
# t_Debye = 276; # K, bulk Nb, book value
# t_Debye = 258; # K, bulk Ta, book value

Tc = 1.2; # K, bulk Al book value
# Tc = 3.4; # K, bulk In book value
# Tc = 9.3; # K, bulk Nb book value
# Tc = 4.5; # K, bulk Ta book value

# Print them out in the beginning, to make sure we're not accidentally using
# wrong values!
print('Using Debye temperature of', t_Debye, 'K');
print('Using Tc of', Tc, 'K');

# The impedance of the metal above Tc is (1 + j) * NORMAL_RESISTANCE
NORMAL_RESISTANCE = 1; # Ohm

###############################################################################
################# Interpolation of reduced gap vs. temperature ###############
###############################################################################

# Enumeration of reduced delta (delta / delta(T=0)) vs. reduced T (T / Tc)
# Universal for weak-coupling superconductor (Nb is example of NOT weak-coupling)

# Modified from old Mathematica code where this was "calculated by Gianluigi somehow and imported"
# This is very similar for different BCS superconductors; I have recalculated it
# in Z:\Lev2.0\Code\Mathematica\TempFitting\bcs_gap_vs_temperature.nb
# which gives a similar, although not identical answer

# We can have this code only run if dtf() is not defined via 
# if not "dtf" in locals()      can also use globals() instead
# However, this can run into issues if a user wants to change the metal properties
# although, as noted above, this shouldn't change the results significantly

if not "dtf" in locals():
    r = 1/TC_OVER_GAP; # The 1.76, easier to have a short name in integral
    
    kappa = t_Debye / (2 * Tc);
    eta = np.log(4 * kappa * TC_OVER_GAP); # Comes from matching T->0 limit analytically
    
    # Function to integrate. x is dummy variable, tau is reduced temperature, delta is reduced gap
    def fff(x, delta, tau):
        return np.tanh(np.sqrt(x**2 * (2 * kappa)**2 + delta**2 * r**2) / (2 * tau)) / \
               np.sqrt(x**2 + delta**2 * r**2 / (2 * kappa)**2);
               
    def ff(delta, tau):
        integral, error = integrate.quad(fff, 0, 1, args = (delta, tau));
        return eta - integral;
    
    
# We now split the reduced temperature into three regions.
# The low and high temperatures are evaluated with the appropriate analytic
# approximations, the middle is further subdivided into three regions.
# Low temperatures are evaluated with 
# Delta ~= 1 - sqrt(2 pi T / 1.76) exp(-1.76/T); see J. Gao's thesis (2.53)
# Middle region is divided into:
#    * 0.16 - 0.50: very flat here, use step of 0.01, guess of 0.95
#    * 0.50 - 0.90: step of 0.005, guess of 0.75
#    * 0.90 - 0.99: step of 0.001, guess of 0.4
# Compare middle region to J. Gao's [46] -- Mühlschlegel 1959, although I don't
# believe this is actually experimental data, unlike what Gao claims.
# For T very close to Tc, delta is well-approximated by the analytic limit
# Delta / Delta(T=0) ~= 1.74 * sqrt(1 - T / Tc) . Use this for T > 0.99 Tc
# See e.g. Tinkham 3.6 for more detail.
# This method is pretty arbitrary, modified from the old code. 
# Takes a few seconds to run.
    
    # Low temps. 
    def delta_low_temp(t):
        return 1 - np.sqrt(2 * np.pi * TC_OVER_GAP * t) * np.exp(-1 / (TC_OVER_GAP * t));
    
    
    # Medium temps
    ts_low = np.arange(0.16, 0.5, 0.01);
    ts_mid = np.arange(0.5, 0.9, 0.005);
    ts_high = np.arange(0.9, 0.99, 0.001)
    ts = np.concatenate([ts_low, ts_mid, ts_high]);
    
    # Use deltas array for initial guess, then overwrite with computed value
    deltas_low = np.ones(len(ts_low)) * 0.95;
    deltas_mid = np.ones(len(ts_mid)) * 0.75;
    deltas_high = np.ones(len(ts_high)) * 0.4;
    deltas = np.concatenate([deltas_low, deltas_mid, deltas_high]);
    
    # Compute delta values for the discrete ts
    for i in range(len(ts)):
        deltas[i] = fsolve(ff, deltas[i], ts[i]);
              
    # Plot for debugging
    if False:         
        plt.plot(ts, deltas);
        plt.title('Computed gap');
        plt.xlabel('T / T_0');
        plt.ylabel(r"\Delta / \Delta(T=0)");
    
    # Fifth-order spline to interpolate between the points. 
    delta_spline = splrep(ts, deltas, k = 5);
    
    # To see what the function and spline look like:
    if False:
        xss = np.linspace(0,1,1000);
        yss = splev(xss, delta_spline);
    
        plt.plot(ts, deltas, 'o', xss, yss);
        
        
    # High temps
    def delta_high_temp(t):
        return 1.74 * np.sqrt(1 - t);
    
    
# Combine:
# dtf(t) is the final function used to give the gap (in units of gap at T = 0)
# as a function of temperature (in units of Tc).
# There are very small discontinuities at 0.16 and 0.99
    def dtf(t):
        def dtf_single(t):
            if(t < 0):
                print("Warning:", t, "< 0! Requested gap at negative temperature.");
                return -1;
            if(t == 0):
                return 1
            elif (t < 0.16):
                return delta_low_temp(t);
            elif(t <= 0.99):
                # Why do we need [()]? Because python is garbage.
                return splev(t, delta_spline)[()];
            elif(t < 1):
                return delta_high_temp(t);
            else:
                assert t >= 1;
                return 0;

               
        # Because python.
        if np.ndim(t) == 0:
            return dtf_single(t);
        return(list(map(dtf_single, t)));
               
    #TODO I was very annoyed by how this was written. Modify it to be more reasonable.
        
        
    if False:
       xss = np.arange(0, 1, 0.001);
       plt.plot(xss, dtf(xss));
   
###############################################################################
#################### Compute complex conductivities using dtf #################
###############################################################################
calc_sigmas = False; # set to true to calculate the sigma tables

# Note that this approach only works in a few approximations.
# In order to compute the surface impedance exactly, we would need to integrate
# the Mattis-Bardeen kernel numerically, see section 2 of Gao's thesis.

# See section 2 in Gao's thesis, which replicates Pöpel '89 (either one)
# in the calculation of the Mattis-Bardeen kernel in cases including 
# hbar omega > 2 Delta (necessary for going through Tc since Delta must go through 0)
# and then gives the forms used here for the relevant approximations.
# Zmuidzinas '12 gives a summary of the situation for lower frequencies. 

#TODO explain variable changes

#TODO these are both divided by normal conductivity
# Real part of complex conductivity
# t is in units of Tc, omega is in units of Delta(T=0)/hbar
# Gao (2.70)
# def sigma1(t, omega):
#     return - integrate.quad(lambda x: (x**2 + omega * x + 1) / np.sqrt(x**2 - 1) /
#            np.sqrt((x + omega)**2 - 1) * np.tanh((x + omega) / 
#            (2 * KB * t * Tc / dtf(t))) , 1 - omega, -1);

#TODO look this probably is not the best way to integrate these, this is just dumb
# application of the general-purpose integrator

max_rel_err = 1e-6 # Converge to this relative error in integration
# To converge on absolute error, set epsabs, and maybe set epsrel to 0
max_iter = 200 # Maximum number of iterations in the integrator

# x is E/Delta, o is hbar omega / Delta, and t is T/Tc / (Delta/Delta(T=0) )
def sigma1(t, o):
    if t == 0:
        return 0;
    # The very low-temperature region is numerically unstable, use approximation
    # within it. To approximate, express tanh'es as exp, find common denominator,
    # remove very small terms (e.g. exp(-x/t)) in denominator.
    if t < 0.15:
        answer1 = integrate.quad(lambda x: 1 / o * (x**2 + o * x + 1) / np.sqrt(x**2 - 1) /
           np.sqrt((x + o)**2 - 1) * 2 * (np.exp(-x / TC_OVER_GAP / t) -
                                          np.exp(-(x + o) / TC_OVER_GAP / t)),
           1, np.inf, epsabs = 0, epsrel = max_rel_err, limit = max_iter);
    else:    
        answer1 = integrate.quad(lambda x: 1 / o * (x**2 + o * x + 1) / np.sqrt(x**2 - 1) /
           np.sqrt((x + o)**2 - 1) * (np.tanh((x + o) / 
           (2 * t * TC_OVER_GAP)) - np.tanh(x/(2 * t * TC_OVER_GAP))) , 1, np.inf,
            epsabs = 0, epsrel = max_rel_err, limit = max_iter); # Converge only on relative error of max_rel_err
        
    if answer1[0] > 0 and answer1[1] / answer1[0] > 1e-6:
        print("Warning: relative error > 1e-6 in sigma1 for  t=", t, "and o =", o);
    if(o <= 2):
        if answer1[0] < 0:
            print("warning: sigma1 < 0 for t =", t, "and o =", o);
        return answer1[0];
    
    #TODO: this next part of the integral differs in sign from that in Gianluigi's code,
    # due to his tanh having argument of just x, not x + o (due to the range of integration
    # and tanh's oddness, this just changes the overall sign). I think this version matches that in the
    # original Mattis&Bardeen though. Hence, confusion. 
    # Note that this might not actually matter for any realistic situations:
    # the zero-temperature o for a 6 GHz Al resonance is 6 GHz * hbar * 2pi / (1.76 * 1.2 K * k_B) = 0.136342
    # This only becomes larger than 2 for temperatures of around 0.999 Tc and above.
    answer2 = integrate.quad(lambda x: 1 / o * (x**2 + o * x + 1) / np.sqrt(x**2 - 1) /
           np.sqrt((x + o)**2 - 1) * np.tanh((x + o) / 
           (2 * t * TC_OVER_GAP)) , 1 - o, -1, epsabs = 0, epsrel = max_rel_err, limit = max_iter);
    if answer2[1] / answer2[0] > 1e-6:
        print("Warning: relative error > 1e-6 in sigma1 for t =", t, "and o =", o);
    if answer1[0] - answer2[0] < 0:
        print("warning: sigma1 < 0 for t =", t, "and o =", o);
    return answer1[0] - answer2[0]; #TODO Why - instead of +?
    
# Imaginary part of complex conductivity
# t is in units of Tc, omega is in units of Delta(T=0)/hbar
# Gao (2.71)
# def sigma2(t, omega):
#     return integrate.quad(lambda x: (x**2 + omega * x + 1) / np.sqrt(1 - x**2) /
#            np.sqrt((x + omega)**2 - 1) * np.tanh((x + omega) / 
#            (2 * KB * t * Tc / dtf(t))) , max(-1, 1 - omega), 1);

# x is E/Delta, o is hbar omega / Delta, and t is T/Tc / (Delta/Delta(T=0) )
def sigma2(t, o):
    answer = integrate.quad(lambda x: 1 / o * (x**2 + o * x + 1) / np.sqrt(1 - x**2) /
           np.sqrt((x + o)**2 - 1) * np.tanh((x + o) / 
           (2 * t * TC_OVER_GAP)) , max(-1, 1 - o), 1, epsabs = 0, epsrel = max_rel_err, limit = max_iter);
    if answer[1] / answer[0] > 1e-6:
        print("Warning: relative error > 1e-6 in sigma2 for t=", t, "and o=", o);
    if answer[0] < 0:
        print("warning: sigma2 < 0 for t =", t, "and o =", o);
    return answer[0];

#TODO this should all eventually be a function, probably
if calc_sigmas:
    ### Write a table of the complex conductivities to a file, so we don't have
    #   to recompute it every time
    #TODO print progress, but better. probably estimate a time too
    #TODO deal with full path when made into function
    sigma1file = 'Sigma1Table.dat';
    sigma2file = 'Sigma2Table.dat';
    
    # What to do if files exist?
    if os.path.exists(sigma1file):
        s1file_flag = input('Warning: %s already exists. Type \'w\' to overwire, \'a\' to append, anything else to cancel: ' % sigma1file);
        s1file_flag = s1file_flag.lower();
        if s1file_flag not in ['w', 'a']:
            exit('Cancelled sigma1 file overwrite');
    if os.path.exists(sigma2file):
        s2file_flag = input('Warning: %s already exists. Type \'w\' to overwire, \'a\' to append, anything else to cancel: ' % sigma2file);
        s2file_flag = s2file_flag.lower();
        if s2file_flag not in ['w', 'a']:
            exit('Cancelled sigma2 file overwrite');
    
    #tf = 15.00; # Around T/Tc = 0.99855
    #tf = 150; # At least in the 200's we already have large error
    #TODO more intelligent meshing for t
    t_array = np.unique(np.concatenate([np.arange(0.005, 0.2, 0.005), np.arange(0.2, 1, 0.01), np.arange(1, 5, 0.2),
                                        np.arange(5, 15, 0.5), np.arange(15, 50, 1), np.arange(50, 150, 5)]));
    
    
    
    # A realistic low-o resonator might be a 3 GHz indium resonator with o = 3 GHz * h / (1.76 * 3.4 K * k_B) = 0.024
    # A 6 GHz Al resonator has o of ~2.5 at 0.999 of Tc.
    # Mesh more finely in regions of larger change. #TODO consider more meshing near 2
    o_array = np.unique(np.concatenate([np.arange(0.01, 0.04, 0.001), np.arange(0.04, 0.1, 0.005),
                              np.arange(0.1, 0.2, 0.01), np.arange(0.2, 1, 0.05), np.arange(1, 1.8, 0.05),
                              np.arange(1.8, 1.9, 0.002), np.arange(1.9, 2.1, 0.001), np. arange(2.1, 2.2, 0.002),
                              np.arange(2.2, 4, 0.05), np.arange(4, 5, 0.1),
                              np.arange(5, 11, 0.5), np.arange(11, 20, 1), np.arange(20, 30, 2),
                              np.arange(30, 75, 5)]));
    

    with open(sigma1file, s1file_flag) as s1f, open(sigma2file, s2file_flag) as s2f:
        for t_ind, t in enumerate(t_array):
            for o in o_array:
                s1f.write(str(t) + '\n' + str(o) + '\n' + str(sigma1(t,o)) + '\n');
                s2f.write(str(t) + '\n' + str(o) + '\n' + str(sigma2(t,o)) + '\n');
            if t_ind % (len(t_array) / 10) < 1:
                # There's surely a better way to do this...
                print(str(int(np.round(t_ind / (len(t_array) / 10)))) + "0% complete.") 
    print("100% complete.")
    
    exit();
            
            
###############################################################################
################ Read in complex conductivities, define impedance #############
###############################################################################           

# Read in the sigma tables, defined as in above section.
### NOTE ### these are defined slightly differently from those in AcrossTcFitting.wl,
#            namely, multiply by o and divide by pi to get those in AcrosTcFitting.wl
# as a function of frequency and temperature
# Equations for computing sigmas can be found in e.g. Gao 2.70, 2.71, or
# Reagor 5.27a/b
sigma_dir = os.path.dirname(os.path.realpath(__file__))

Sigma1In = np.loadtxt(sigma_dir + '\\Sigma1Table.dat')
Sigma2In = np.loadtxt(sigma_dir + '\\Sigma2Table.dat')

Sigma1In = np.reshape(Sigma1In, (int(len(Sigma1In) / 3), 3))
Sigma2In = np.reshape(Sigma2In, (int(len(Sigma2In) / 3), 3))

# Only works for regular grids, so need all x's and y's in ascending order,
# then need 2D array of all zs.
ts1 = np.unique(Sigma1In[:,0])
os1 = np.unique(Sigma1In[:,1])
ss1 = np.reshape(Sigma1In[:,2],(len(ts1), len(os1)))

ts2 = np.unique(Sigma2In[:,0])
os2 = np.unique(Sigma2In[:,1])
ss2 = np.reshape(Sigma2In[:,2],(len(ts2), len(os2)))


# default degree is 3. Only works for regular grids. call as rbs(x,y)[0,0]
# or rbs.ev(x,y)[0]
# rbs1/2 is the spline for sigma1/2, x is related to temperature, y is
# related to frequency
#TODO better description
# ss1[t index][o index]
# rbs1_1 is the linear spline, to be used in the region near o/gap = 2
# since in this region, the derivative is discontinuous and higher-order
# splines actually introduct large errors.
# rbs1_3 is the cubic spline, to be used in all other regions as it's otherwise
# a better fit.
rbs1_1 = RectBivariateSpline(ts1, os1, ss1, ky = 1)
rbs1_3 = RectBivariateSpline(ts1, os1, ss1, ky = 3)

def rbs1_combined(t, o):
    if o < 2 and t < 0.05:
        return 0;
    if o < 1.85 or o > 2.15:
        return rbs1_3.ev(t, o);
    else:
        return rbs1_1.ev(t, o);
    
rbs2 = RectBivariateSpline(ts2, os2, ss2)


# Takes forever to run (never actually let it finish).
# My guess is that this is so slow because it doesn't optimise using the fact
# that it's over a regular grid. Leaving it in for now for reference. 
#Sigma1Int = interp2d(Sigma1In[:,0], Sigma1In[:,1], Sigma1In[:,2],
#                     kind = 'cubic')
#Sigma2Int = interp2d(Sigma2In[:,0], Sigma2In[:,1], Sigma2In[:,2],
#                     kind = 'cubic')

# Surface impedance.
# o is hbar omega / Delta(T) #TODO this is wrong, I think it's hbar omega * Delta(0)
# t is T/Tc
# n is nu:
# -1/2 for dirty limit, -1/3 for clean limit
# Rn is normal-state resistivity, in Ohms
#TODO check that this is right
# See Nam [7] Eq. (1.11) for this expression. Recall Xn = Rn for normal metal 
# (excepting anomalous conductivity, easy to modify code for that)
def zsf(t, o, n, Rn):
    def zsf_single(t):
        if t < 1:
            return (1 + 1j) * ( (rbs1_combined(t / dtf(t), o / dtf(t)) - \
                1j * rbs2.ev(t / dtf(t), o / dtf(t))) )**n;
        
        # Thick film, local limit
        #return np.sqrt(1j * MU0 * 2 * np.pi * f0 / ((rbs1.ev(t / dtf(t), o / dtf(t)) -
        #          1j * rbs2.ev(t / dtf(t), o / dtf(t))) * dtf(t)));
        else:
            return (1 + 1j) * Rn;
               
    # Because python.
    if np.ndim(t) == 0:
        return zsf_single(t);  
    return(list(map(zsf_single, t)));

# Surface reactance
def xsf(t, o, n, Rn):
    return np.imag(zsf(t, o, n, Rn));
    
# Surface resistance. Arguments same as above.
def rsf(t, o, n, Rn):
    return np.real(zsf(t, o, n, Rn));

###############################################################################
###################### Bardeen-Mattis fitting functions. ######################
###############################################################################
 
    
# Returns frequency shift from f0 scaled by f0. t in mK, FitTc in K, alpha, n dimensionless, Rn in Ohms.
def f_model(t, params):
     FitTc = params['Tc'];
     alpha = params['alpha'];
     n = params['n'];
     f0 = params['f0'];
     Rn = params['Rn'];
     XS0 = xsf(ALMOST_ZERO, 2 * np.pi * f0 * HBAR / (KB * FitTc / TC_OVER_GAP), n, Rn)
     return alpha * (xsf(t / 1000 / FitTc, f0 / (KB / (HBAR * 2 * np.pi) * 
                            FitTc / TC_OVER_GAP), n, Rn) - XS0) / (2 * XS0)
 
# Returns expected Qint. t in mK, FitTc in K, alpha, n and Q0 dimensionless, Rn in Ohms.
def q_model(t, params):
     FitTc = params['Tc'];
     alpha = params['alpha'];
     Q0 = params['Q0']
     n = params['n'];
     f0 = params['f0'];
     Rn = params['Rn'];
     XS0 = xsf(ALMOST_ZERO, 2 * np.pi * f0 * HBAR / (KB * FitTc / TC_OVER_GAP), n, Rn)
     return np.reciprocal((alpha / XS0 * rsf(t / 1000 / FitTc, 
                         2 * np.pi * f0 * HBAR / (KB * FitTc / TC_OVER_GAP), 
                         n, Rn)) + 1/Q0)
     
        
# Fits frequency shift and Qint changes due to thermal quasiparticles using the
#    Bardeen-Mattis theory. Uses lmfit.
# dat_file is read in via read_in_csv, see for format
# tc_guess is in K
# alpha_guess, Q0_guess are just numbers
# n = -1/2 for dirty superconductor (local), -1/3 for clean (extreme anomalous). 
# See Gao's thesis for a description of the regimes.
# If the film is much thinner than penetration depth, n = -1, with some other
#     changes, see Zmuidzinas' review paper [4] eq. (11) and (12).
# Q/freq_drop_start and Q/freq_drop_end are indices of Q/freq data to drop. If 
#    end is 0, it is fixed to be the end of the array.
# r: residuals from the Q fit are r times more important than those of the freq fit
# print_results prints extracted fit parameters, which requires simulated pmag and
#    the lambda used to calculate the pmag (really, we only need the ratio of the two).
# fit_Q and fit_freq tell us whether we want to fit by the Q, by the frequency,
#    or to to fit both.
# fix_tc, fix_alpha, fix_Q0, fix_Rn tell us whether to fix these parameters or allow them to vary.
# error_method: string used to specify where we get the errors for y axes, values:
#    * 'scale': error is some constant times the value on the y axis
#    * 'from_file': read in errors from data_file, presumably from circlefit
#    * 'constant': error is some constant value for every point
def BardeenMattisFit(data_file, tc_guess, alpha_guess, Q0_guess, n, 
           freq_drop_start = 0, freq_drop_end = 0,
           Q_drop_start = 0, Q_drop_end = 0, r = 1, Rn_guess = 1, print_results = True,
           fit_Q = True, fit_freq = True,
           fix_tc = False, fix_alpha = False, fix_Q0 = False, fix_Rn = False,
           error_method = 'scale'):
    
    # Check that we're fitting at least one dataset.
    if not fit_Q and not fit_freq:
        print('Please fit by at least Q or freq.');
        return None;
    
    # Read in all the data from csv
    f0, temperatures_f, frequencies, frequency_errors,temperatures_Q, Qs, Q_errors = \
        read_in_csv(data_file, freq_drop_start, freq_drop_end, Q_drop_start,
                    Q_drop_end);
        
    # Print out fitting temperature ranges:
    if(fit_freq):
        print('Fitting frequencies from %.1f mK to %.1f mK' % (temperatures_f[0],
          temperatures_f[-1]));
    if(fit_Q):
        print('Fitting Qs from %.1f mK to %.1f mK' % (temperatures_Q[0],
          temperatures_Q[-1]));
    
    # Convert frequencies to scaled shift from f0. Remember to fix errors too.
    freq_shifts = (f0 - frequencies) / f0;
    freq_shift_errors = frequency_errors / f0;
    if error_method == 'scale':
        freq_shift_errors = np.sign(freq_shift_errors)*1e1; #TODO explain
        Q_errors = Qs / 1e5;
    elif error_method == 'constant':
        freq_shift_errors = np.sign(freq_shift_errors)*1e1;
        Q_errors = np.sign(Q_errors)*1e2;
    elif error_method != 'from_file':
        print('Please choose error_method as \'scale\', \'constant\', or \'from_file\'!');
        return None;
        

    # LMfit style of fitting
    params = Parameters()
    params.add('Tc', value = tc_guess, vary = not fix_tc, min = 0.01);#, max = 3.4);
    params.add('alpha', value = alpha_guess, vary = not fix_alpha, min = 0);
    # Don't allow Q0 to vary for only a frequency fit. #TODO
    params.add('Q0', value = Q0_guess, min = 1e-1, vary = fit_Q and not fix_Q0);
    params.add('n', value = n, vary = False); # always fixed at the moment
    params.add('f0', value = f0, vary = False); # store f0 in params
    params.add('Rn', value = Rn_guess, vary = not fix_Rn);
    
    # t_f and t_Q are different since we probably cut off some lower temps for Q
    # Weigh Q errors by r more than frequency errors if fitting both.
    def objective(params, t_f, freq_shifts, freq_shift_errors, t_Q, Qs, Q_errors, r):
        if fit_Q and fit_freq:
            return np.concatenate(((f_model(t_f, params) - freq_shifts) / freq_shift_errors,
                              (q_model(t_Q, params) - Qs) / Q_errors * r), axis = 0);
        elif fit_Q:
             return (q_model(t_Q, params) - Qs) / Q_errors ;
        else: # fit_freq
            assert fit_freq;
            return (f_model(t_f, params) - freq_shifts) / freq_shift_errors;
    
    
    return minimize(objective, params, args = (temperatures_f, freq_shifts,
                           freq_shift_errors, temperatures_Q, Qs, Q_errors, r));


###############################################################################
################################ Other functions ##############################
###############################################################################
    
# Reads in data from a .csv file             
# data_file is a .csv with columns, including (but not limited to)
#    * 'temperature (mK)' -- temperature in mK
#    * 'fr'               -- frequency in Hz from circle fit
#    * 'fr_stderr'        -- standard errors on frequency (in Hz) from circle fit    
#    * 'Qi'               -- internal Q from circle fit
#    * 'Qi_stderr'        -- standard error on internal Q from circle fit
# Returns a tuple of (f0 (frequency at lowest temperature), temperatures_f, 
#    frequencies, frequency_errors, temperatures_Q, Qs, Q_errors)
# the frequency/Q variables span indices from freq/Q_drop_start to freq/Q_drop_end
def read_in_csv(data_file, freq_drop_start = 0, freq_drop_end = 0,
                Q_drop_start = 0, Q_drop_end = 0):
    # Read in the data
    df = pandas.read_csv(data_file);
    file_data = np.transpose(np.array([df['temperature (mK)'].values, df['fr'].values,
            df['fr_stderr'].values, df['Qi'].values, df['Qi_stderr'].values]));
    # Sort all data by temperature
    file_data = file_data[file_data[:,0].argsort()]
    f0 = file_data[0, 1];
    
    # Separate out the different quantities into different variables.
    temperatures_f = file_data[:,0];
    temperatures_Q = temperatures_f.copy();
    frequencies = file_data[:,1];
    frequency_errors = file_data[:,2];
    Qs = file_data[:,3];
    Q_errors = file_data[:,4];
    
    # Deal with dropping some frequencies. Don't normally need to, but made 
    # available in case it's desired. 
    if freq_drop_end == 0:
        freq_drop_end = -frequencies.shape[0];  
    frequencies = frequencies[freq_drop_start : -freq_drop_end];
    frequency_errors = frequency_errors[freq_drop_start : -freq_drop_end];
    temperatures_f = temperatures_f[freq_drop_start : -freq_drop_end];
    
    # Deal with dropping some Q's. Generally, will want to drop Q's at lower
    # temperature as these are limited by non-temperature-dependent sources
    # combined with TLS losses.
    if Q_drop_end == 0: 
        Q_drop_end = -Qs.shape[0] # length of array 
    Qs = Qs[Q_drop_start : -Q_drop_end];
    Q_errors = Q_errors[Q_drop_start : -Q_drop_end];
    temperatures_Q = temperatures_Q[Q_drop_start : -Q_drop_end];
    
    return (f0, temperatures_f, frequencies, frequency_errors, 
            temperatures_Q, Qs, Q_errors);
  
# Plot results, compare data with model
# data_file is read in with read_in_csv, see it for format
# params contains Tc, alpha, and n. Again, see BardeenMattisFit
# same for freq/Q_drop_start/end, although note that this is for plot, not for fit
# plot_area: plots curves both from params and params 2 and shades area between
#    * this requires params2, will print error and return None if params2 is None
# plot_freq/Q: plot the frequency/Q figure
# plot_Q_log: make the Q plot semilog-y. Not implemented for freq as it's not as useful
# plot_Q_reciprocal: plot 1/Q instead of Q
# plot_residuals: plots residuals instead of freq/Q plot
#    * only makes residuals plot for whichever plot is asked for by plot_freq/Q
#    * if plot_area is True, plots two sets of residuals on one graph
#    * residuals are defined as the data - the fit
# freq/Q_temp_start/stop (mK): manually set the temperature domain of the plot,
#   in case we want to e.g. plot both data and the fit function over a wider range
#   of temperatures than what the data has
def plot_results(data_file, params, freq_drop_start = 0, freq_drop_end = 0,
                 Q_drop_start = 0, Q_drop_end = 0, plot_area = False, params2 = None,
                 plot_freq = True, plot_Q = True, plot_Q_log = True, plot_Q_reciprocal = False,
                 plot_residuals = False, plot_title = '',
                 # If we want to override the default temperature domain of the plot. in mK
                 freq_temp_start = 0, freq_temp_stop = 0, Q_temp_start = 0, Q_temp_stop = 0):
    if plot_area and not params2:
        print("Need second set of parameters to plot an area!");
        return None;
        
    # Read in all the data from csv
    f0, temperatures_f, frequencies, frequency_errors,temperatures_Q, Qs, Q_errors = \
        read_in_csv(data_file, freq_drop_start, freq_drop_end, Q_drop_start,
                    Q_drop_end);
        
    # Evaluate models at these points:
    if not freq_temp_start:
        freq_temp_start = temperatures_f[0];
    if not freq_temp_stop:
        freq_temp_stop = temperatures_f[-1];
    if not Q_temp_start:
        Q_temp_start = temperatures_Q[0];
    if not Q_temp_stop:
        Q_temp_stop = temperatures_Q[-1];
        
    freq_domain = np.linspace(freq_temp_start, freq_temp_stop, num = 1000)
    Q_domain = np.linspace(Q_temp_start, Q_temp_stop, num = 1000)
    
    # Convert frequencies to scaled shift from f0. Remember to fix errors too.
    freq_shifts = (f0 - frequencies) / f0;
    freq_shift_errors = frequency_errors / f0;
        
    ### Some plot settings copied from yseamQiPlot
    fontsize = 8
    tickfontsize = 8
    linewidth = 1 #0.5 #1.0
    markersize = 5.0
    size_inches = [3.5, 2.625]
    
    fig_freq = None;
    fig_Q = None;
    
    ### Figure plot
    if(plot_freq):
        fig_freq = plt.figure() # The argument to this is not the title but the figure label!
        fig_freq.set_size_inches(size_inches)
        
        ax_freq = plt.gca()
        #ax1.yaxis.set_label_position("right")
        ax_freq.set_xlabel('Temperature (mK)', fontsize=fontsize)
        ax_freq.tick_params(axis='both', which='both', direction='in', 
                        labelsize=tickfontsize)
        
        #plt.xlim([1.5e-6, 1.5e5])
        #plt.ylim([1e2, 1e10])
    
        # If you use scatter, there will be problems with the y scaling
        # Multuply by -f0 to get to absolute frequency, divide by 1000 to get it in kHz
        # Base plots
        if not plot_residuals:
            ax_freq.set_ylabel('Frequency shift (kHz)', fontsize=fontsize)
            ax_freq.plot(temperatures_f, freq_shifts * -f0 / 1000, 'r', marker = '^',  
                 ls = '', markersize = markersize, label='data') 
            ax_freq.plot(freq_domain, f_model(freq_domain, params) * -f0 / 1000, 'k-', 
                 linewidth = linewidth, label='fit')
        else:
            ax_freq.set_ylabel('Frequency shift residuals (kHz)', fontsize=fontsize)
            #ax_freq.plot(temperatures_f, (freq_shifts - f_model(temperatures_f, params)) * \
            #              -f0 / 1000, marker = '*', ls = '', 
            #              markersize = markersize, label = 'residuals'); 
            ax_freq.errorbar(temperatures_f, (freq_shifts - f_model(temperatures_f, params)) * \
                          -f0 / 1000, yerr = freq_shift_errors * -f0/1000);
                #TODO
        # Additional plots if plot_area is on:
        if plot_area and not plot_residuals: 
            #ax_freq.fill_between(freq_domain, f_model(freq_domain, params)* -f0 / 1000,
            #         f_model(freq_domain, params2) * -f0 / 1000, color = '#ff7878');
            ax_freq.plot(freq_domain, f_model(freq_domain, params2) * -f0 / 1000, 'k--', 
                 linewidth = linewidth, label='fit2')
            #ax_freq.legend(['Frequency fit', 'Q fit'])
        elif plot_area and plot_residuals:
            ax_freq.plot(temperatures_f, (freq_shifts - f_model(temperatures_f, params2)) * \
                          -f0 / 1000, marker = 'o', ls = '', 
                          markersize = markersize, label = 'residuals');  

        ax_freq.set_title(plot_title);
        fig_freq.tight_layout()
        #plt.ticklabel_format(axis='y', style='sci', scilimits = (0,0))
        
    ### Q plot:
    if plot_Q:
        fig_Q = plt.figure() # The argument to this is not the title but the figure label!
        fig_Q.set_size_inches(size_inches)
        
        ax_Q = plt.gca()
        #ax1.yaxis.set_label_position("right")
        ax_Q.set_xlabel('Temperature (mK)', fontsize=fontsize)
        ax_Q.tick_params(axis='both', which='both', direction='in', 
                        labelsize=tickfontsize)
 
        #plt.xlim([1.5e-6, 1.5e5])
        #plt.ylim([1e2, 1e10])
        
    
        # If you use scatter, there will be problems with the y scaling
        if plot_Q_log:
            plot_function = ax_Q.semilogy
            coeff = 1.;
            suffix = '';
        else:
            plot_function = ax_Q.plot
            coeff = 10.**6;
            suffix = ' (millions)';
            
        if plot_Q_reciprocal:
            # Need the 0. because for some reason Python has integer division....?
            Q_data = (coeff + 0.) / Qs
            Q_model = (coeff + 0.) / q_model(Q_domain, params)
            prefix = '{Q_{\mathrm{int}}}^{-1}';
        else:
            Q_data = Qs / coeff
            Q_model = q_model(Q_domain, params) / coeff
            prefix = 'Q_{\mathrm{int}}';
            
        if plot_residuals:
            plot_function(temperatures_Q, Q_data - q_model(temperatures_Q, params), 
                      color='#73bbff', marker = 'o', ls = '', markersize = markersize, label='data')  
            infix = ' residuals';
        else:
            plot_function(temperatures_Q, Q_data, color='#73bbff', marker = 'o', ls = '', 
                          markersize = markersize, label='data')  
            # This next line can be important -- matplotlib seems to randomly
            # change the exact behaviour of autoscaling between versions in
            # undocumented ways, resulting in empty plots.
            #axisLimits = ax_Q.get_ylim()
            ax_Q.set_autoscale_on(True)
            #ax_Q.set_ylim(axisLimits[0], axisLimits[1])
            plot_function(Q_domain, Q_model, 'k-',
                         linewidth = linewidth, label='fit')
            infix = '';
            
        if plot_area:
            if plot_Q_reciprocal:
                Q_model2 = (coeff + 0.) / q_model(temperatures_Q, params2)
            else:
                Q_model2 = q_model(temperatures_Q, params2) / coeff
                
                
        if plot_area and not plot_residuals:
            #ax_Q.fill_between(Q_domain, q_model(Q_domain, params)/coeff,
            #                 q_model(Q_domain, params2)/coeff, color = '#8e8eff',
            #                 clip_on = True)
                
            plot_function(Q_domain, Q_model2, 'k--',
                         linewidth = linewidth, label='fit2', clip_on = True)
            plt.ylim(ax_Q.get_ylim()[0], ax_Q.get_ylim()[1])
            #ax_Q.legend(['Frequency fit', 'Q fit'])
        elif plot_area and plot_residuals:
            plot_function(temperatures_Q, Q_model2, 'k-',
                         linewidth = linewidth, label='fit2')
            
        # Create y-axis label based on whether it's a lin/log plot and residual or not and reciprocal or not
        ax_Q.set_ylabel('$' + prefix + infix + suffix + '$', fontsize=fontsize)
        ax_Q.set_title(plot_title);
        fig_Q.tight_layout();
        
    return fig_freq, fig_Q
    
###############################################################################  
############################### Main code #####################################
###############################################################################  

data_file = 'fitresults_temp.csv';

# Initial guesses
tc_guess = 1.2 # in K
alpha_guess = 5e-5
Q0_guess = 1e6
# n = -1/2 for dirty superconductor (local), -1/3 for clean (extreme anomalous). 
n = -1./2.

# Weigh Q fit sameish as freq fit
r = 1;

# HFSS-simulated conductor participation, assuming lambda = 50 nm
pmag = 7.5e-5
simulation_lambda = 50e-9 # m

# Do we do fits for only freq, only Q, both?
fit_just_freq = True
fit_just_Q = True
fit_both = False


# Do we save the plot for fitting frequency, Q?
save_freq = False;
save_Q = False;
save_both = False;

freqfit_freqplot_save_file = 'freq_fit_freq_plotplot.pdf';
freqfit_Qplot_save_file = 'freq_fit_Qi_plot.pdf'
Qfit_freqplot_save_file = 'Qi_fit_freq_plot.pdf';
Qfit_Qplot_save_file = 'Qi_fit_Qi_plot.pdf';
bothfit_freqplot_save_file = 'Qi_and_freq_fit_freq_plot.pdf';
bothfit_Qplot_save_file = 'Qi_and_freq_fit_Qi_plot.pdf';

### Frequency alone fitting ###
if fit_just_freq:
    print('Fitting only frequency...');
    # drop first drop_start elements and last drop_end elements (0 for full data)
    # drop_end should be >= 0
    freq_drop_start = 1
    freq_drop_end = 0
    freq_fit_results = BardeenMattisFit(data_file, tc_guess, alpha_guess, Q0_guess,
                        n, freq_drop_start = freq_drop_start, freq_drop_end = freq_drop_end,
                        fit_Q = False, fit_freq = True);
    fig_freqfit_freqplot, fig_freqfit_Qplot = plot_results(data_file, 
                               freq_fit_results.params, plot_residuals = False,
                               plot_Q_log = True, plot_title = 'Fit only frequency')
    report_fit(freq_fit_results);
    
    calculated_lambda_freq = freq_fit_results.params['alpha'].value / (pmag / simulation_lambda)
    calculated_lambda_freq_error =  freq_fit_results.params['alpha'].stderr /       \
                                        (pmag / simulation_lambda);
    print("Lambda from frequency: %.2f ± %.2f nm\n" % (calculated_lambda_freq * 1e9, 
                                               calculated_lambda_freq_error * 1e9))
    
    #if save_freq:
        #plt.savefig(freq_save_file, transparent=True)

### Q alone fitting ###
if fit_just_Q:
    print('Fitting only Q...')
    # drop first drop_start elements and last drop_end elements (0 for full data)
    # drop_end should be >= 0
    Q_drop_start = 1
    Q_drop_end = 0
    
    Q_fit_results = BardeenMattisFit(data_file, tc_guess, alpha_guess, Q0_guess,
                  n, Q_drop_start = Q_drop_start, Q_drop_end = Q_drop_end,
                  fit_Q = True, fit_freq = False)
    fig_Qfit = plot_results(data_file, Q_fit_results.params, plot_Q_log = True,
                            plot_title = 'Fit only Q')    
    report_fit(Q_fit_results);
    
    calculated_lambda_Q = Q_fit_results.params['alpha'].value / (pmag / simulation_lambda)
    calculated_lambda_Q_error =  Q_fit_results.params['alpha'].stderr /       \
                                        (pmag / simulation_lambda);
    print("Lambda from Q: %.2f ± %.2f nm\n" % (calculated_lambda_Q * 1e9, 
                                               calculated_lambda_Q_error * 1e9))

    #if save_Q:
      #  plt.savefig(Q_save_file, transparent=True)

### Fitting both ###
if fit_both:
    print('Fitting both frequency and Q...')
    Q_drop_start = 30
    Q_drop_end = 0
    
    both_fit_results = BardeenMattisFit(data_file, tc_guess, alpha_guess, Q0_guess, n,
                    Q_drop_start = Q_drop_start, Q_drop_end = Q_drop_end, r=r);
    fig_bothfit = plot_results(data_file, both_fit_results.params, 
                               plot_Q_log = True, plot_title = 'Fit both')    
    report_fit(both_fit_results);
    
    calculated_lambda_both = both_fit_results.params['alpha'].value / (pmag / simulation_lambda)
    calculated_lambda_both_error =  both_fit_results.params['alpha'].stderr /       \
                                        (pmag / simulation_lambda);
    print("Lambda from both: %.2f ± %.2f nm\n" % (calculated_lambda_both * 1e9, 
                                               calculated_lambda_both_error * 1e9))
   # if save_both:
       # plt.savefig(both_save_file, transparent=True)

### Plot a given set of parameters
if False:
    params = Parameters()
    params.add('Tc', value = 0.9);
    params.add('alpha', value = 1.09e-4);
    params.add('Q0', value = 1e6);
    params.add('n', value = -0.5); # always fixed at the moment
    params.add('f0', value = 9.670574e+09)
    plot_results(data_file, params, plot_Q_log = True, plot_residuals = False, Q_drop_start = 6)  
    
    
# Plot local limit, show both f and Q fits on each plot
#plot_results(data_file, freq_fit_results.params, params2 = Q_fit_results.params, 
#             plot_area = True, plot_title = 'Solid is freq fit, dashed is Q fit')





#################################### DEBUGGING ###############################
# Ignore this section
### Debugging code to make plots of impedance
if False:
    FitTc = 1.2;
    f0 = 9e9;
    n = -1/2;
    ooo=f0 * HBAR * 2 * np.pi * TC_OVER_GAP / (KB  * FitTc)
    
    temps = np.arange(1100,1200, 0.1);
    
    xsfs = xsf(temps / 1000 / FitTc, f0 / (KB / (HBAR * 2 * np.pi) * 
                                FitTc / TC_OVER_GAP), n);
    rsfs = rsf(temps / 1000 / FitTc, f0 / (KB / (HBAR * 2 * np.pi) * 
                                FitTc / TC_OVER_GAP), n);
    
    plt.plot(temps, xsfs);
    
    
    rsf(1-1e-2, f0 / (KB / (HBAR * 2 * np.pi) * FitTc / TC_OVER_GAP), n)
    
    
    
    
    
    freqs = np.arange(0, 200e9, 1e9);
    tttt = 0.9;
    
    xsfs = xsf(tttt, freqs / (KB / (HBAR * 2 * np.pi) *  
                             FitTc / TC_OVER_GAP), n);
    rsfs = rsf(tttt, freqs / (KB / (HBAR * 2 * np.pi) *  
                             FitTc / TC_OVER_GAP), n);
    
    plt.plot(freqs, xsfs);
    
    
    
    
    
    
    
    
    
    temps = np.arange(0, 1300, 1);
    params = Parameters()
    params.add('Tc', value = 1.2);
    params.add('alpha', value = 1.e-4);
    params.add('Q0', value = 1e6);
    params.add('n', value = -0.5); # always fixed at the moment
    params.add('f0', value = 9e+09)
    freqs = f_model(temps, params);
    Qs = q_model(temps, params);
    
    plt.plot(temps,freqs);
    
    plt.plot(temps,Qs);
    
    ##########################################################################
    ################## Plots for comparison to papers ########################
    ##########################################################################
    
    # Biondi and Garfunkel 1959, Fig. 4 = Pöpel 1989, Fig. 3 (b)
    n = -1/2
    temps = np.concatenate((np.arange(0.005, 1, 0.005), [0.999]));
    plt.figure()
    for o in [0.64, 1.66, 2.46, 3.08, 3.63, 3.91]:
        # I think they used 1.7 instead of 1.76 here
        plt.plot(temps, [rsf(t, o*TC_OVER_GAP, n, 1) for t in temps])
        #plt.plot(temps, [np.real(zsf_combined(t, o*TC_OVER_GAP, n)) for t in temps])
        plt.scatter(temps, [np.real(-np.exp(-1j * np.pi / 2. * (1 - n))*((sigma1(t / dtf(t), o*TC_OVER_GAP / dtf(t)) - \
                1j * sigma2(t / dtf(t), o*TC_OVER_GAP / dtf(t))))**n)*np.sqrt(2) for t in temps])
        
   
    

    
    
    FitTc = 1.2;
    f0 = 9e9;
    n = -1/2;
    
    temps = np.arange(0.95, 1, 0.0005);
    
    sigma1s = rbs1_combined.ev(temps / dtf(temps), f0 * HBAR * 2 * np.pi * TC_OVER_GAP / (KB  * FitTc) / dtf(temps));
    sigma2s = rbs2.ev(temps / dtf(temps), f0 * HBAR * 2 * np.pi * TC_OVER_GAP / (KB  * FitTc) / dtf(temps));
    
    plt.plot(temps, sigma1s);
