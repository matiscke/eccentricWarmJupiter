""" small tools and equations for typicall exoplanet inferences.
Martin Schlecker, 2020

schlecker@mpia.de
"""
import numpy as np
from astropy import units as u
from astropy import constants as c

try:
    from astropy.timeseries import LombScargle
except ModuleNotFoundError:
    # the timeserie module moved at astropy 3.1 -> 3.2
    from astropy.stats import LombScargle

au2m = 1.496e11

def format(paramName):
    """ Get Latex-formatted labels for some quantities."""
    if paramName.startswith('log '):
        paramName = paramName[4:]
        log = True
    else:
        log = False

    labels = {
    'P_p1' : '$P \, [\mathrm{d}]$',
    't0_p1' :'$t_0\, [\mathrm{BJD}]$',
    'r1_p1' :'$r_1$',
    'r2_p1' :'$r_2$',
    'b_p1' : '$b$',
    'p_p1' : '$R_\mathrm{P}/R_\star$',
    'sesinomega_p1' : '$\sqrt{e}\sin(\omega \cdot \pi/180)$',
    'secosomega_p1' : '$\sqrt{e}\cos(\omega \cdot \pi/180)$',
    'ecc' : 'e',
    'omega' : r'$\omega \, [\mathrm{deg}]$',
    'rho' : r'$ \rho $',
    'q1_TESSERACT+TESS' : '$q_\mathrm{1, TESS}$',
    'q2_TESSERACT+TESS' : '$q_\mathrm{2, TESS}$',
    'u1_TESSERACT+TESS' : '$u_\mathrm{1, TESS}$',
    'u2_TESSERACT+TESS' : '$u_\mathrm{2, TESS}$',
    'sigma_w_TESSERACT+TESS' : '$\sigma_\mathrm{w, TESS}$',
    'mflux_TESSERACT+TESS' :   '$M_\mathrm{TESS}$',
    'mdilution_TESSERACT+TESS' : '$D_\mathrm{TESS}$',
    'GP_sigma_TESSERACT+TESS' :  '$GP_\mathrm{\sigma, TESS}$',
    'GP_timescale_TESSERACT+TESS' :  '$GP_\mathrm{t, TESS}$',
    'q1_CHAT+i' :  '$q_\mathrm{1, CHAT}$',
    'q2_CHAT+i' :  '$q_\mathrm{2, CHAT}$',
    'u1_CHAT+i' :  '$u_\mathrm{1, CHAT}$',
    'u2_CHAT+i' :  '$u_\mathrm{2, CHAT}$',
    'sigma_w_CHAT+i' : '$\sigma_\mathrm{w, CHAT}$',
    'mflux_CHAT+i' :   '$M_\mathrm{CHAT}$',
    'mdilution_CHAT+i' :'$D_\mathrm{CHAT}$',
    'metallicity': '[Fe/H]',
    'm': '$M_\mathrm{P} \, [\mathrm{M_\oplus}]$',
    'a': '$a \, [\mathrm{au}]$',
    'r': '$R_\mathrm{P} \, [\mathrm{R_{Jup}}]$',
    'r_rEarth': '$R_\mathrm{P} \, [\mathrm{R_\oplus}]$',
    'K_p1' : '$K \, [\mathrm{m/s}]$',
    'mu_FEROS' :'$\mu_\mathrm{FEROS}\, [\mathrm{m/s}]$',
    'sigma_w_FEROS' :'$\sigma_\mathrm{FEROS}\, [\mathrm{m/s}]$',
    'rv_intercept' :'$b_\mathrm{RV}\, [\mathrm{m/s}]$',
    'rv_slope' : '$m_\mathrm{RV}\, [\mathrm{m/s}]$',
    'rv_quad' : '$q_\mathrm{RV}\, [\mathrm{m/s}]$',
    'incl_p1' : 'i [deg]',
    'radius_rj_p1' : '$R_\mathrm{P} \, [\mathrm{R_{Jup}}]$',
    'mass_mj_p1' : '$M_\mathrm{P} \, [\mathrm{M_{Jup}}]$',
    'semimajor_au_p1' : '$a \, [\mathrm{au}]$',
    'rho_kgm3_p1' : r'$\rho_\mathrm{P} \, [\mathrm{kg \, m^{-3}}]$'
    }
    try:
        label = labels[paramName]
    except KeyError:
        label = paramName

    if log:
        label = 'log ' + label
    return label


def label(originalLabel):
    """reformat some instrument labels for legends and stuff"""
    labels = {
        'TESSERACT+TESS' : 'TESS',
        'CHAT+i' : 'CHAT'
    }

    try:
        label = labels[originalLabel]
    except KeyError:
        label = originalLabel

    return label

def photPlotParams():
    """ return a dictionary with preferred matplotlib parameters for photometry plots."""
    return {
        'alpha' : 0.66,
        'fmt' : '.',
        'elinewidth' : .5,
        'ms' : 1.,
        'color' : 'black'
    }

def r_of_theta(theta, a, e=0.):
    """ compute orbital distance from true anomaly for an eccentric orbit."""
    r = a*(1-e**2)/(1+e*np.cos(theta))
    return r


def Teq(L, a, albedo=0., emissivity=1., beta=1.):
    """ compute the instantaneous equilibrium temperature of a planet at orbital
    distance r.

    See equation 3 in Kaltenegger+2011.
    
    Parameters
    ----------
    L : float
        stellar luminosity [L_sol]
    a : float
        semi-major axis [au]
    albedo : float
        planetary albedo
    emissivity : float
        broadband thermal emissivity (usually ~ 1)
   beta : float
       fraction of the planet surface that re-radiates the absorbed flux. 
       beta = 1 for fast rotators and beta ~ 0.5 for tidally locked planets 
       without oceans or atmospheres (Kaltenegger & Sasselov 2011).

    Returns
    --------
   teq : float
        instantaneous equilibrium temperature
    """
    T0 = 278.5  # Teq of Earth for zero albedo [K]

    # teq = Teff*((1. - albedo)/(beta*emissivity))**(1./4.)*np.sqrt(0.5*Rstar/r)
    teq = T0 * ((1. - albedo) * L / (beta * emissivity * a ** 2)) ** (1. / 4.)
    return teq


def avgTeq(L, a, e=0., albedo=0., emissivity=1., beta=1.):
    """compute the time-averaged equilibrium temperature as in Mendez+2017.

    This uses eqn 16 in Méndez A, Rivera-Valentín EG (2017) Astrophys J 837:L1.
    https://doi.org/10.3847/2041-8213/aa5f13"

    Parameters
    ----------
    L : float
        stellar luminosity [L_sol]
    a : float
        semi-major axis [au]
    e : float
        eccentricity
    albedo : float
        planetary albedo
    emissivity : float
        broadband thermal emissivity (usually ~ 1)
    beta : float
        fraction planet surface that re-radiates absorbed flux

    Returns
    -------
    Teq : float
        time-averaged equilibrium temperature 
    
    
    """
    T0 = 278.5  # Teq of Earth for zero albedo [K]
    avgTeq = T0*(((1 - albedo)*L)/(beta*emissivity*a**2))**(1/4)*(1 - e**2/16 - (15/1024)*e**4)
    return avgTeq

def get_GLS(t, rv, rv_error):
    """ compute the Generalized Lomb-Scargle periodogram.

    Notes
    -----
    The irregularly-sampled data enables frequencies much higher than the
    average Nyquist frequency.
    """
    ls = LombScargle(t, rv, rv_error, fit_mean=True)
    frequency, power = ls.autopower() # minimum_frequency=f_min, maximum_frequency = f_max) ### automated f limits from baseline and Nyquist factor
    i_peak = np.argmax(power)
    peakFrequency = frequency[i_peak]
    peakPower = power[i_peak]
    FAP = ls.false_alarm_probability(peakPower, method='bootstrap')
    return peakFrequency, peakPower, FAP, ls


def get_Tcirc(a, Mstar, Rp, Mp, e=0, Qp=1e6):
    """ estimate orbit circularization timescale.

    Parameters
    -----------
    a : float
        sma in au
    Mstar : float
        stellar mass in Msol
    Rp : float
        planetary radius in Rjup
    Mp : float
        planetary mass in Mjup
    e : float
        eccentricity


    Equation 2 in Adams FC, Laughlin G (2006),
    Long‐Term Evolution of Close Planets Including the Effects of Secular Interactions.
    Astrophys J 649:1004–1009. https://doi.org/10.1086/506145
    """
    F = lambda e: 1 + 6*e**2
    tau = 4/63*Qp*np.sqrt((a*c.au)**3/(c.G*Mstar*c.M_sun))*Mp*c.M_jup/(Mstar*c.M_sun)*(a*c.au/(Rp*c.R_jup))**5*(1-e**2)**(13/2)/F(e)
    return tau.to(u.yr)
