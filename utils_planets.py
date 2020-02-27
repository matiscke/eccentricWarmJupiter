import numpy as np
import juliet
# https://exoplanetarchive.ipac.caltech.edu/docs/poet_calculations.html

G = 6.67408e-11 # m3 kg-1 s-2
solarrad2m = 6.957e8 # solar radii in meters
solarmass2kg = 1.9891e30 # solar mass in kg
earthrad2m = 6.371e6 # earth radii in meters
jupiterrad2m = 69.911e6 # jupiter radii in meters
earthmass2kg = 5.972e24 # earth mass in kg
jupitermass2kg = 1.898e27 # jupiet mass in kg
ergs2solarlum = 3.839e33 # erg s-1 = 1L solar
AU2m = 149600000000 # 1 AU = 149600000000m

def get_planetaryparams(dataset, results, mass_star=None, radius_star=None, teff_star=None, lum_star=None, vsini_star=None, \
                        albedo=None, emissivity=None):
    """
    This takes in the results object and generates as many of the planetary parameters as possible (given the stellar parameters as input) 

    :param dataset: (juliet object)
        An object containing all the information regarding the data to be fitted, including options of the fit. 
        Generated via juliet.load().

    :param results: (juliet object)
        An object containing all the information regarding the fit to the data, including the posteriors specficially. 
        This will be modified.
        Generated via juliet.fit(dataset).

    :param mass_star: (array, float/int) -- units in solar mass
        Defines the mass of the star. It can either be 
        (1) a user-inputed distribution (i.e from isochrone-gaia fitting);
        (2) an array such as [mass_star, mass_star_error] which will produce a Gaussian distribution;
        (3) it can be a single value which is simply mass_star.
    
    :param radius_star: (array, float/int) -- units in solar radii
        Defines the radius of the star. It can either be 
        (1) a user-inputed distribution (i.e from isochrone-gaia fitting);
        (2) an array such as [radius_star, radius_star_error] which will produce a Gaussian distribution;
        (3) it can be a single value which is simply radius_star.

    :param teff_star: (array, float/int) -- units in Kelvin
        Defines the effective temperature of the star. It can either be
        (1) a user-inputed distribution;
        (2) an array such as [temp_star, temp_star_error] which will produce a Gaussian distribution;
        (3) it can be a single value which is simply temp_star.

    :param lum_star: (array, float/int) -- units in solar luminosity
        Defines the luminosity of the star. It can either be
        (1) a user-inputed distribution;
        (2) an array such as [lum_star, lum_star_error] which will produce a Gaussian distribution;
        (3) it can be a single value which is simply lum_star.
        Note that if this is not provided then the luminosity is still calculated if the radius and effective temp are given.

    :param vsini_star: (float/int) -- units in km/s
        Defines the rotational velocity of the star. It can only be a single value which is simply vsini_star.

    :param albedo: (dict)
        Defines the albedo of each planet, where albedo={'1': 0.0, '2':0.1, ...}
        If not provided, then all albedos for the planets are set to 0.0.

    :param emissivity: (dict)
        Defines the emissivity of each planet, where emissivity={'1': 1.0, '2':0.9, ...}
        If not provided, then all emissivitivites for the planets are set to 1.0.
    """
    transiting_planets = dataset.numbering_transiting_planets # array of planet indexes that transit
    rv_planets = dataset.numbering_rv_planets # array of planet indexes that have RVs
    num_samps = len(results.posteriors['posterior_samples']['unnamed'])

    if mass_star is not None:
        if mass_star.__class__.__name__ in ('list', 'numpy.ndarray', 'ndarray'):
            if np.size(mass_star) == 2: # [value,err] was given as input, assume Gaussian distribution
                m_star = np.random.normal(loc=mass_star[0], scale=mass_star[1], size=num_samps)
            else: # user input was a distribution
                m_star = np.random.choice(mass_star, size=num_samps, replace=False)
        else:
            m_star = mass_star
    if radius_star is not None:
        if radius_star.__class__.__name__ in ('list', 'numpy.ndarray', 'ndarray'):
            if np.size(radius_star) == 2: # [value,err] was given as input, assume Gaussian distribution
                r_star = np.random.normal(loc=radius_star[0], scale=radius_star[1], size=num_samps)
            else: # user input was a distribution
                r_star = np.random.choice(radius_star, size=num_samps, replace=False)
        else:
            r_star = radius_star
    if teff_star is not None:
        if teff_star.__class__.__name__ in ('list', 'numpy.ndarray', 'ndarray'):
            if np.size(teff_star) == 2: # [value,err] was given as input, assume Gaussian distribution
                t_star = np.random.normal(loc=teff_star[0], scale=teff_star[1], size=num_samps)
            else: # user input was a distribution
                t_star = np.random.choice(teff_star, size=num_samps, replace=False)
        else:
            t_star = teff_star
    if lum_star is not None:
        if lum_star.__class__.__name__ in ('list', 'numpy.ndarray', 'ndarray'):
            if np.size(lum_star) == 2: # [value,err] was given as input, assume Gaussian distribution
                l_star = np.random.normal(loc=lum_star[0], scale=lum_star[1], size=num_samps)
            else: # user input was a distribution
                l_star = np.random.choice(lum_star, size=num_samps, replace=False)
        else:
            l_star = lum_star
    else: # user did not input it
        if radius_star is not None and teff_star is not None:
            l_star = get_Lstar(r_s=r_star*solarrad2m, teff=t_star)

    ## obtain a/R given the stellar density (rho)
    if 'rho' in results.posteriors['posterior_samples'].keys() or 'rho' in dataset.priors.keys():
        try:
            rho = results.posteriors['posterior_samples']['rho']
        except KeyError: # rho was a fixed parameter then
            rho = dataset.priors['rho']['hyperparameters']
            # if dataset.priors['rho']['distribution'] == 'fixed':
            #     rho = dataset.priors['rho']['hyperparameters']
            # elif dataset.priors['rho']['distribution'] == 'normal':
            #     rho = np.random.normal(loc=dataset.priors['rho']['hyperparameters'][0], scale=dataset.priors['rho']['hyperparameters'][1],size=num_samps)
            # else:
            #     rho = dataset.priors['rho']['hyperparameters'][0]
        for i_transit in transiting_planets:
            try:
                P = results.posteriors['posterior_samples']['P_p{}'.format(i_transit)]
            except KeyError:
                P = dataset.priors['P_p{}'.format(i_transit)]['hyperparameters']

            a = ((rho*G*((P*24.*3600.)**2))/(3.*np.pi))**(1./3.)
            results.posteriors['posterior_samples']['a_p{}'.format(i_transit)] = a

    ## obtain the ecc and omega for all planets so that it is in the posteriors
    for i_planet in np.unique(np.append(rv_planets, transiting_planets)):
        if 'secosomega_p{}'.format(i_planet) in dataset.priors.keys(): # sqrt(e)cosomega, sqrt(e)sinomega
            try:
                secosomega = results.posteriors['posterior_samples']['secosomega_p{}'.format(i_planet)]
                sesinomega = results.posteriors['posterior_samples']['sesinomega_p{}'.format(i_planet)]
            except:
                secosomega = dataset.priors['secosomega_p{}'.format(i_planet)]['hyperparameters']
                sesinomega = dataset.priors['sesinomega_p{}'.format(i_planet)]['hyperparameters']

            results.posteriors['posterior_samples']['ecc_p{}'.format(i_planet)] = secosomega**2 + sesinomega**2
            results.posteriors['posterior_samples']['omega_p{}'.format(i_planet)] = np.arccos(secosomega/np.sqrt(results.posteriors['posterior_samples']['ecc_p{}'.format(i_planet)]))*180./np.pi
        elif 'ecosomega_p{}'.format(i_planet) in dataset.priors.keys(): # ecosomega,esinomega
            try:
                ecosomega = results.posteriors['posterior_samples']['ecosomega_p{}'.format(i_planet)]
                esinomega = results.posteriors['posterior_samples']['esinomega_p{}'.format(i_planet)]
            except:
                ecosomega = dataset.priors['ecosomega_p{}'.format(i_planet)]['hyperparameters']
                esinomega = dataset.priors['esinomega_p{}'.format(i_planet)]['hyperparameters']
            results.posteriors['posterior_samples']['ecc_p{}'.format(i_planet)] = np.sqrt(ecosomega**2+esinomega**2)
            results.posteriors['posterior_samples']['omega_p{}'.format(i_planet)] = np.arctan2(esinomega,ecosomega)
        else: # ecc,omega
            if 'ecc_p{}'.format(i_planet) not in results.posteriors['posterior_samples'].keys(): # it was most likely fixed so adapt that value
                results.posteriors['posterior_samples']['ecc_p{}'.format(i_planet)] = dataset.priors['ecc_p{}'.format(i_planet)]['hyperparameters']
            if 'omega_p{}'.format(i_planet) not in results.posteriors['posterior_samples'].keys(): # it was most likely fixed so adapt that value
                results.posteriors['posterior_samples']['omega_p{}'.format(i_planet)] = dataset.priors['omega_p{}'.format(i_planet)]['hyperparameters']*np.pi/180.

        # In the case that ecc/omega were single values, we need to convert them into arrays of size num_samps
        if results.posteriors['posterior_samples']['ecc_p{}'.format(i_planet)].__class__.__name__ in ('list', 'numpy.ndarray', 'ndarray'):
        	continue
        else:
        	if results.posteriors['posterior_samples']['ecc_p{}'.format(i_planet)] in (0, 0.0): 
        		results.posteriors['posterior_samples']['ecc_p{}'.format(i_planet)] = np.ones(num_samps)*results.posteriors['posterior_samples']['ecc_p{}'.format(i_planet)]
        		results.posteriors['posterior_samples']['omega_p{}'.format(i_planet)] = np.ones(num_samps)*90. # It does not matter which value of omega
        	else:
        		results.posteriors['posterior_samples']['ecc_p{}'.format(i_planet)] = np.ones(num_samps)*results.posteriors['posterior_samples']['ecc_p{}'.format(i_planet)]
        		results.posteriors['posterior_samples']['omega_p{}'.format(i_planet)] = np.ones(num_samps)*results.posteriors['posterior_samples']['omega_p{}'.format(i_planet)]

    ## deal with the albedo and emissivity if not given and set albedo=0.0 and emissivity=1.0
    if albedo is None:
        albedo = {}
        for i_planet in np.unique(np.append(rv_planets, transiting_planets)):
            albedo[i_planet] = 0.0
    if emissivity is None:
        emissivity = {}
        for i_planet in np.unique(np.append(rv_planets, transiting_planets)):
            emissivity[i_planet] = 1.0    

    for i_transit in transiting_planets:
        try:
            P = results.posteriors['posterior_samples']['P_p{}'.format(i_transit)]
        except KeyError:
            P = dataset.priors['P_p{}'.format(i_transit)]['hyperparameters']

        ## obtain b and p
        if 'r1_p{}'.format(i_transit) in results.posteriors['posterior_samples'].keys() or 'r1_p{}'.format(i_transit) in dataset.priors.keys():
            try:
                r1 = results.posteriors['posterior_samples']['r1_p{}'.format(i_transit)]
            except KeyError:
                r1 = dataset.priors['r1_p{}'.format(i_transit)]['hyperparameters']
            try:
                r2 = results.posteriors['posterior_samples']['r2_p{}'.format(i_transit)]
            except KeyError:
                r2 = dataset.priors['r2_p{}'.format(i_transit)]['hyperparameters']
            b,p = juliet.utils.reverse_bp(r1,r2,pl=0,pu=1)
            results.posteriors['posterior_samples']['b_p{}'.format(i_transit)] = b
            results.posteriors['posterior_samples']['p_p{}'.format(i_transit)] = p

        ## obtain the radius of the planet -- requires r_star
        if radius_star is not None and ('p_p{}'.format(i_transit) in results.posteriors['posterior_samples'].keys() or 'p_p{}'.format(i_transit) in dataset.priors.keys()):
            try:
                p = results.posteriors['posterior_samples']['p_p{}'.format(i_transit)]
            except KeyError:
                p = dataset.priors['p_p{}'.format(i_transit)]['hyperparameters']

            results.posteriors['posterior_samples']['radius_m_p{}'.format(i_transit)] = p * r_star*solarrad2m
            results.posteriors['posterior_samples']['radius_re_p{}'.format(i_transit)] = results.posteriors['posterior_samples']['radius_m_p{}'.format(i_transit)]/earthrad2m
            results.posteriors['posterior_samples']['radius_rj_p{}'.format(i_transit)] = results.posteriors['posterior_samples']['radius_m_p{}'.format(i_transit)]/jupiterrad2m

        ## obtain the inclination
        results.posteriors['posterior_samples']['incl_p{}'.format(i_transit)] = get_inclination(aR=results.posteriors['posterior_samples']['a_p{}'.format(i_transit)],\
                                                                                                b=results.posteriors['posterior_samples']['b_p{}'.format(i_transit)],\
                                                                                                e=results.posteriors['posterior_samples']['ecc_p{}'.format(i_transit)],\
                                                                                                omega=results.posteriors['posterior_samples']['omega_p{}'.format(i_transit)])
        
        ## obtain the transit duration
        results.posteriors['posterior_samples']['transittime_p{}'.format(i_transit)] = get_transittime(period=P, aR=results.posteriors['posterior_samples']['a_p{}'.format(i_transit)],\
                                                                                                    p=results.posteriors['posterior_samples']['p_p{}'.format(i_transit)],\
                                                                                                    b=results.posteriors['posterior_samples']['b_p{}'.format(i_transit)])
        ## obtain the semimajor axis -- requires r_star
        if radius_star is not None:
            results.posteriors['posterior_samples']['semimajor_au_p{}'.format(i_transit)] = get_semimajoraxis(aR=results.posteriors['posterior_samples']['a_p{}'.format(i_transit)],\
                                                                                                        r_s=r_star)
        ## obtain the RM effect -- requires r_star and vsin_star
        if radius_star  is not None and vsini_star is not None:
            results.posteriors['posterior_samples']['rm_p{}'.format(i_transit)] = get_rm(vsini=vsini_star, \
                                                                                        r_p=results.posteriors['posterior_samples']['radius_rj_p{}'.format(i_transit)],\
                                                                                        r_s=r_star)
        ## obtain the Teq -- requires the teff_star
        if teff_star is not None:
            results.posteriors['posterior_samples']['teq_p{}'.format(i_transit)] = get_teq(teff=t_star, aR=results.posteriors['posterior_samples']['a_p{}'.format(i_transit)],\
                                                                                        albedo=albedo[i_transit],emissivity=emissivity[i_transit])
        ## obtain the isolation -- requires lum_star and semimajor axis 
        # either user gave l_star or l_star was computed via r_star and t_star
        if (lum_star is not None or (radius_star is not None and teff_star is not None)) and 'semimajor_au_p{}'.format(i_transit) in results.posteriors['posterior_samples'].keys():
            results.posteriors['posterior_samples']['insolation_p{}'.format(i_transit)] = get_insolation(l_s=l_star, \
                                                                                        a_au=results.posteriors['posterior_samples']['semimajor_au_p{}'.format(i_transit)])

    for i_rv in rv_planets:
        try:
            P = results.posteriors['posterior_samples']['P_p{}'.format(i_rv)]
        except KeyError:
            P = dataset.priors['P_p{}'.format(i_rv)]['hyperparameters']
        try:
            K = results.posteriors['posterior_samples']['K_p{}'.format(i_rv)]
        except KeyError:
            K = dataset.priors['K_p{}'.format(i_rv)]['hyperparameters']

        ## obtain the msini (or mass if inclination is available)
        if mass_star is not None:
            results.posteriors['posterior_samples']['msini_kg_p{}'.format(i_rv)] = get_msini(period=P, \
                                                                                            K=K, \
                                                                                            m_s=m_star, \
                                                                                            e=results.posteriors['posterior_samples']['ecc_p{}'.format(i_rv)])
            results.posteriors['posterior_samples']['msini_me_p{}'.format(i_rv)] = results.posteriors['posterior_samples']['msini_kg_p{}'.format(i_rv)] / earthmass2kg
            results.posteriors['posterior_samples']['msini_mj_p{}'.format(i_rv)] = results.posteriors['posterior_samples']['msini_kg_p{}'.format(i_rv)] / jupitermass2kg

            if 'incl_p{}'.format(i_rv) in results.posteriors['posterior_samples'].keys():
                i = results.posteriors['posterior_samples']['incl_p{}'.format(i_rv)]
                fact = 1./np.sin(i*np.pi/180.)
                results.posteriors['posterior_samples']['mass_kg_p{}'.format(i_rv)] = fact*results.posteriors['posterior_samples']['msini_kg_p{}'.format(i_rv)]
                results.posteriors['posterior_samples']['mass_me_p{}'.format(i_rv)] = fact*results.posteriors['posterior_samples']['msini_me_p{}'.format(i_rv)]
                results.posteriors['posterior_samples']['mass_mj_p{}'.format(i_rv)] = fact*results.posteriors['posterior_samples']['msini_mj_p{}'.format(i_rv)]

        ## obtain the gravity and planet density (if mass and radius are available)
        if 'mass_kg_p{}'.format(i_rv) in results.posteriors['posterior_samples'].keys() and 'radius_m_p{}'.format(i_rv) in results.posteriors['posterior_samples'].keys():
            results.posteriors['posterior_samples']['gravity_p{}'.format(i_rv)] = get_gravity(m_p=results.posteriors['posterior_samples']['mass_kg_p{}'.format(i_rv)],\
                                                                                            r_p=results.posteriors['posterior_samples']['radius_m_p{}'.format(i_rv)])

            results.posteriors['posterior_samples']['rho_kgm3_p{}'.format(i_rv)] = get_planetrho(m_p=results.posteriors['posterior_samples']['mass_kg_p{}'.format(i_rv)],\
                                                                                        r_p=results.posteriors['posterior_samples']['radius_m_p{}'.format(i_rv)])
            results.posteriors['posterior_samples']['rho_gcm3_p{}'.format(i_rv)] = results.posteriors['posterior_samples']['rho_kgm3_p{}'.format(i_rv)] * 1000. / (100.)**3
        ## obtain semi major axis is not already from the transit parameters
        if 'semimajor_au_p{}'.format(i_rv) not in results.posteriors['posterior_samples'].keys() and mass_star:
            results.posteriors['posterior_samples']['semimajor_au_p{}'.format(i_rv)] = get_semimajoraxis_from_period(period=P, m_s=m_star)

        ## obtain the Teq if not already there from the transit parameters
        if 'teq_p{}'.format(i_rv) not in results.posteriors['posterior_samples'].keys() and 'semimajor_au_p{}'.format(i_rv) in results.posteriors['posterior_samples'].keys() and radius_star and teff_star:
            results.posteriors['posterior_samples']['teq_p{}'.format(i_rv)] = get_teq_from_semimajoronly(teff=t_star, r_s=r_star, \
                                                                    semimajor= results.posteriors['posterior_samples']['semimajor_au_p{}'.format(i_rv)],\
                                                                    albedo=albedo[i_rv],emissivity=emissivity[i_rv])
        
        ## obtain the insolation if not already there from the transit parameters
        if 'insolation_p{}'.format(i_rv) not in results.posteriors['posterior_samples'].keys():
            if (lum_star or (radius_star and teff_star)) and 'semimajor_au_p{}'.format(i_rv) in results.posteriors['posterior_samples'].keys():
                results.posteriors['posterior_samples']['insolation_p{}'.format(i_rv)] = get_insolation(l_s=l_star, \
                                                                                        a_au=results.posteriors['posterior_samples']['semimajor_au_p{}'.format(i_rv)])


def get_inclination(aR, b, e=0, omega=90):
    # applicable for transits only
    # aR = the ratio of the planet-star distance to the stellar radius
    # b = the impact parameter of the orbit
    return np.arccos(b/aR * ((1+e*np.sin(omega))/(1-e**2))) * 180./np.pi

def get_semimajoraxis(aR, r_s):
    # applicable for transits only
    # aR = the ratio of the planet-star distance to the stellar radius
    # r_s [solar radius] = radius of star
    # returns semi-major axis in AU
    return aR * r_s * solarrad2m / AU2m

def get_transittime(period, aR, p, b):
    # applicable for transits only
    # period [days] = orbital period
    # aR = the ratio of the planet-star distance to the stellar radius
    # p = the planet-to-star radius ratio (Rp/Rs)
    # b = the impact parameter of the orbit
    # the transit time is returned in hours
    num1, num2 = (1+p)**2, b**2
    num = num1+num2
    dom = 1 - (b/aR)**2
    term1, term2 = 1./aR, np.sqrt(num/dom)
    Td = period/np.pi * np.arcsin(term1*term2)
    return Td * 24 #hours

def get_rm(vsini, r_p, r_s):
    # applicable for transits only
    # vsini [km/s] = rotational velocity of star
    # r_p [jupiter radii] = mass of planet
    # r_s [solar radii] = radius of star
    # returns RM effect in m/s
    # Adopted from eqn 6 in http://adsabs.harvard.edu/abs/2007ApJ...655..550G
    return 52.8 * (vsini/5.) * (r_p)**2 / (r_s)**2

def get_teq(teff, aR, albedo=0.0, emissivity=1.0):
    # applicable for transits only
    # teff [K] = effective temparature of the star
    # aR = the ratio of the planet-star distance to the stellar radius
    # returns teq which is the planet's equilibrium temperature in K
    teq = teff * ((1. - albedo)/emissivity)**(1./4.)*np.sqrt(0.5/aR)
    return teq

def get_Lstar(r_s, teff):
    # https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_constant
    # r_s [m] = radius of star
    # teff [K] = effective temparature of the star
    # the luminosity returned is in solar luminosity
    sigma = 5.6704e-5 #erg cm-2 s-1 K-4
    L = 4*np.pi*(r_s*100.)**2*sigma*teff**4 # ergs s-1
    return L/ergs2solarlum

def get_insolation(l_s, a_au):
    # for transit (and rv if transit wasn't there)
    # l_s [solar lum] = luminosity of the star
    # a_au [AU] = planet-star distance
    # returns insolation in earth isolation
    return l_s / (a_au**2)    

def get_msini(period, K, m_s, e=0):
    # applicable for rvs only
    # m*sini = (P/2piG)*1/3 * K * M^2/3 * (1-e^2)^1/2
    # K [m/s] = semi-amplitude of RV signal
    # m_s [solarmass] = mass of star
    # returns msini of the planet in kg
    msini = ((period*24*3600)/(2*np.pi*G))**(1./3) * K * (m_s*solarmass2kg)**(2./3) * np.sqrt(1-e**2)
    return msini#/np.sin(i*np.pi/180.)

def get_semimajoraxis_from_period(period, m_s):
    # applicable for rvs if the planet does not transit
    # http://orbitsimulator.com/cmc/a2.html
    # period [days] = orbital period
    # m_s [solarmass] = mass of star
    # returns semi-major axis in AU
    fact = 1./(4*np.pi**2)
    return (fact * (period*24*3600)**2 * G * m_s*solarmass2kg)**(1./3) / AU2m
    
def get_teq_from_semimajoronly(teff, r_s, semimajor, albedo=0.0, emissivity=1.0):
    # applicable for rvs if the planet does not transit
    # a_au [AU] = planet-star distance
    # r_s [solar radii] = radius of star
    aR = semimajor*AU2m/(r_s*solarrad2m)
    teq = teff * ((1. - albedo)/emissivity)**(1./4.)*np.sqrt(0.5/aR)
    return teq    

def get_gravity(m_p, r_p):
    # applicable for planets with transits and rv
    # m_p [kg] = mass of planet
    # r_p [m] = radius of planet
    # returns g = GM/r^2 [m/s^2]
    return G*m_p/r_p**2

def get_planetrho(m_p, r_p):
    # applicable for planets with transits and rv
    # m_p [kg] = mass of planet
    # r_p [m] = radius of planet
    # returns the planetary density in kg/m3
    volume = (4./3.*np.pi*r_p**3)
    return m_p / volume

