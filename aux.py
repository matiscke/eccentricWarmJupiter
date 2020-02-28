""" small tools and equations for typicall exoplanet inferences.
Martin Schlecker, 2020

schlecker@mpia.de
"""
import numpy as np

au2m = 1.496e11

def r_of_theta(theta, a, e=0.):
    """ compute orbital distance from true anomaly for an eccentric orbit."""
    r = a*(1-e**2)/(1+e*np.cos(theta))
    return r

def Teq(r, Rstar, Teff, albedo=0., emissivity=1., beta=1.):
    """ compute the instantaneous equilibrium temperature of a planet at orbital
    distance r. See equation 3 in Kaltenegger+2011.
    
   beta : float
       fraction of the planet surface that re-radiates the absorbed flux. 
       beta = 1 for fast rotators and beta ~ 0.5 for tidally locked planets 
       without oceans or atmospheres (Kaltenegger & Sasselov 2011).
    """
    teq = Teff*((1. - albedo)/(beta*emissivity))**(1./4.)*np.sqrt(0.5*Rstar/r)
    return teq
