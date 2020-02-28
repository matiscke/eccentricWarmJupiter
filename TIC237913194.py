import numpy as np
import matplotlib.pyplot as plt
import juliet
import plots

datafolder = '/Users/schlecker/WINE/TIC237913194/data/'
out_folder = 'test12'
instruments_lc = ['TESSERACT+TESS', 'CHAT+i']
instruments_rv = ['FEROS']
colors_rv = ['orangered','cornflowerblue', 'purple', 'forestgreen']

# target parameters
Mstar = 1.03 # solar mass
Mstar_err = .06
Rstar = 1.088 # solar raddi
Rstar_err = .012
Teff = 5788

solarrad2m = 6.957e8 # solar radii in meters

"""
Define the priors
"""
priors = {}
params = {
# planet 1
'P_p1' : ['normal', [15.16, 0.2]],
't0_p1' : ['normal', [2458319.17, 0.2]],
'r1_p1' : ['uniform', [0.,1.]],
'r2_p1' : ['uniform', [0.,1.]],
'sesinomega_p1' : ['uniform', [-1, 1]],
'secosomega_p1' : ['uniform', [-1, 1]],

# Star
'rho' : ['normal', [1120,110]],

# TESS
'q1_TESSERACT+TESS' : ['uniform', [0., 1.]],
'q2_TESSERACT+TESS' : ['uniform', [0., 1.]],
'sigma_w_TESSERACT+TESS' : ['loguniform', [1e-5,1e5]],
'mflux_TESSERACT+TESS' : ['normal', [0.0,0.1]],
'mdilution_TESSERACT+TESS' : ['fixed', 1.0],
'GP_sigma_TESSERACT+TESS' : ['loguniform', [1e-7, 1e-2]],
'GP_timescale_TESSERACT+TESS' : ['loguniform', [1e-3, 3]],

# CHAT+i
'q1_CHAT+i' : ['uniform', [0., 1.]],
# 'q2_CHAT+i' : ['uniform', [0., 1.]],
'sigma_w_CHAT+i' : ['loguniform', [1e-5,1e5]],
'mflux_CHAT+i' : ['normal', [0.0,0.1]],
'mdilution_CHAT+i' : ['fixed', 1.0],

# # RV planetary
'K_p1' : ['uniform', [0.,0.1]], # it is given in km/s

# RV FEROS
'mu_FEROS' : ['uniform', [-10,40]],
'sigma_w_FEROS' : ['loguniform', [1e-5,1.]],

# long-term trend
# 'rv_intercept' : ['normal', [0.0,100]],
# 'rv_slope' : ['normal', [0.0,1.0]],
# 'rv_quad' : ['normal', [0.0,1.0]]
}

# transform priors into a format juliet understands
priors = {}
for name in params.keys():
    priors[name] = {'distribution' : params[name][0], 'hyperparameters' : params[name][1]}


"""
Read data
"""
# read photometry
gp_times_lc = {}
times_lc, fluxes, fluxes_error = {},{},{}
for inst in instruments_lc:
    times_lc[inst], fluxes[inst], fluxes_error[inst] = np.loadtxt(datafolder+inst.lower()+'.lc.dat',
                                                                  unpack=True, usecols=(0,1,2))
    if inst == 'TESSERACT+TESS':
        # mask outliers in TESS photometry
        i_out = [992, 1023, 1036, 1059, 1060, 1061, 1078, 1082, 1083, 1084, 1602]
        times_lc[inst] = np.delete(times_lc[inst], i_out)
        fluxes[inst] = np.delete(fluxes[inst], i_out)
        fluxes_error[inst] = np.delete(fluxes_error[inst], i_out)

        # include GPs for TESS`
        gp_times_lc[inst] = times_lc[inst]

    # plot photometry
    plt.scatter(times_lc[inst], fluxes[inst],s=1)
    plt.title(inst)
    plt.show()


# read RVs
times_rv, rvs, rvs_error = {},{},{}
for inst in instruments_rv:
    times_rv[inst], rvs[inst], rvs_error[inst] = np.loadtxt(datafolder+inst.lower()+'.rv.dat',
                                                            unpack=True, usecols=(0,1,2))

# Load dataset into juliet
dataset = juliet.load(
    priors=priors, t_lc=times_lc, y_lc=fluxes, yerr_lc=fluxes_error,
    t_rv=times_rv, y_rv=rvs, yerr_rv=rvs_error,
    GP_regressors_lc =gp_times_lc,
    out_folder=out_folder, verbose=True)

# Fit and absorb results into a juliet.fit object:
results = dataset.fit(use_dynesty=True, n_live_points = 500, ecclim=0.7,
                      dynesty_nthreads=7)

# plot posteriors
fig = plots.plot_cornerPlot(results, params)
plt.show()
fig.savefig(out_folder + '/cornerPosteriors.pdf')

# Plot the photometry with fit:
fig, ax = plots.plot_photometry(dataset, results)
plt.show()
fig.savefig(out_folder + '/photometryFitted.pdf')
pass


""" Compute additional quantities 
"""
# current best-fit params
a = 16785120000.
e = 0.575
fig, ax = plots.plot_Teq_theta(a, e, Rstar*solarrad2m, Teff)
