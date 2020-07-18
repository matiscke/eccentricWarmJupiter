import numpy as np
import matplotlib.pyplot as plt
import juliet
import plots
import latextable
import pickle

try:
    from popsyntools import plotstyle
except ModuleNotFoundError:
    print('module "popsyntools" not found. Skipping plot styles therein.')

datafolder = 'data/'

# out_folder = 'out/39_tess+chat+feros+GP'
# out_folder = 'out/40_tess+chat+feros+GP+linearTrend'
# out_folder = 'out/41_tess+chat+lcogt+feros+GP'
out_folder = 'out/42_tess+chat+lcogt+feros+GP' # same but number



# constrain Rp/Rs
pl=0.0
pu=0.5


if 'GP' in out_folder and not 'noGP' in out_folder:
    GP = True # include Gaussian Process regressor for TESS data
else:
    GP = False

# instruments_lc = []
# instruments_lc = ['TESSERACT+TESS']
instruments_lc = ['TESSERACT+TESS', 'CHAT+i', 'LCOGT']
outlierIndices = [992, 1023, 1036, 1059, 1060, 1061, 1078, 1082, 1083, 1084, 1602] # outliers in TESS light curve
instruments_rv = ['FEROS']
# instruments_rv = ['FEROS', 'CORALIE']
colors_rv = ['orangered', 'cornflowerblue', 'purple', 'forestgreen']

# Stellar parameters for TIC237913194
Mstar = 1.03 # solar mass
Mstar_err = .06
Rstar = 1.088 # solar raddi
Rstar_err = .012
Teff = 5788
Lstar = 1.196


solarrad2m = 6.957e8 # solar radii in meters

def get_priors(GP=True):
    """ Define the priors. """
    params = {
    # planet 1

    'P_p1' : ['normal', [15.16, 0.2]],
    # 'P_p1' : ['uniform', [1, 30]],
    # 'P_p1' : ['loguniform', [.1, 30]],
    # 'P_p1' : ['fixed', 1.],
    # 'P_p2' : ['uniform', [1, 30]],

    't0_p1' : ['normal', [2458319.17, 0.2]],
    # 't0_p1' : ['uniform', [2458325.32623291, 2458449.32623291]],
    # 't0_p1' : ['uniform', [2458670.0, 2458700.0]],
    # 't0_p1' : ['fixed', 1.],
    # 't0_p2' : ['uniform', [2458325.32623291, 2458449.32623291]],


    # 'r1_p1' : ['uniform', [0.,1.]],
    # 'r2_p1' : ['uniform', [0.,1.]],
    'p_p1' : ['uniform', [0., .5]],
    'b_p1' : ['uniform', [0., 1.5]],

    'sesinomega_p1' : ['uniform', [-1, 1]],
    # 'sesinomega_p1' : ['fixed', 0.],
    'secosomega_p1' : ['uniform', [-1, 1]],
    # 'secosomega_p1' : ['fixed', 0.],
    # 'sesinomega_p2' : ['uniform', [-1, 1]],
    # 'secosomega_p2' : ['uniform', [-1, 1]],
    # 'ecc_p1' : ['fixed', 0.], # fix to circular orbit
    # 'omega_p1' : ['fixed', 90.], # fix to circular orbit
    # 'ecc_p2' : ['fixed', 0.], # fix to circular orbit
    # 'omega_p2' : ['fixed', 90.], # fix to circular orbit

    # Star
    'rho' : ['normal', [1120,110]],

    # TESS
    'q1_TESSERACT+TESS' : ['uniform', [0., 1.]],
    'q2_TESSERACT+TESS' : ['uniform', [0., 1.]],
    'sigma_w_TESSERACT+TESS' : ['loguniform', [1e-5,1e5]],
    'mflux_TESSERACT+TESS' : ['normal', [0.0,0.1]],
    'mdilution_TESSERACT+TESS' : ['fixed', 1.0],
    'GP_sigma_TESSERACT+TESS' : ['loguniform', [1e-8, 5e-4]],
    'GP_timescale_TESSERACT+TESS' : ['loguniform', [1e-4, 2]],

    # CHAT+i
    'q1_CHAT+i' : ['uniform', [0., 1.]],
    ##### 'q2_CHAT+i' : ['uniform', [0., 1.]],
    'sigma_w_CHAT+i' : ['loguniform', [1e-5,1e5]],
    'mflux_CHAT+i' : ['normal', [0.0,0.1]],
    'mdilution_CHAT+i' : ['fixed', 1.0],


    # LCOGT
    'q1_LCOGT' : ['uniform', [0., 1.]],
    ##### 'q2_LCOGT' : ['uniform', [0., 1.]],
    'sigma_w_LCOGT' : ['loguniform', [1e-5,1e5]],
    'mflux_LCOGT' : ['normal', [0.0,0.1]],
    'mdilution_LCOGT' : ['fixed', 1.0],

    # RV planetary
    'K_p1' : ['uniform', [140., 260.]], # prior based on RV-only fit (fit No 33)
    # 'K_p1' : ['fixed', 0.], # no-planet-case
    # 'K_p2' : ['uniform', [0., 1000.]],


    # RV FEROS
    'mu_FEROS' : ['uniform', [-30, 30]], # prior based on RV-only fit (fit No 33)
    # 'sigma_w_FEROS' : ['loguniform', [0.01, 1e3]],
    'sigma_w_FEROS' : ['loguniform', [1., 1e2]], # prior based on RV-only fit (fit No 33)

    # RV CORALIE
    # 'mu_CORALIE' : ['uniform', [-1000, 1000]],
    # 'sigma_w_CORALIE' : ['loguniform', [0.01, 1e3]],

    # long-term trend
    # 'rv_intercept' : ['normal', [0.0,10000]],
    # 'rv_slope' : ['normal', [0.0,1.0]],
    # 'rv_quad' : ['normal', [0.0,1.0]]
    }

    if not GP:
        try:
            del params['GP_sigma_TESSERACT+TESS']
            del params['GP_timescale_TESSERACT+TESS']
        except KeyError:
            pass

    # transform priors into a format juliet understands
    priors = {}
    for name in params.keys():
        priors[name] = {'distribution' : params[name][0],
                        'hyperparameters' : params[name][1]}


    return priors, params

def read_photometry(datafolder, plotPhot=False, outlierIndices=None):
    """ Read photometry from files. 
    
    Format has to be <Instrumentname>.lc.dat
    """
    gp_times_lc = {}
    times_lc, fluxes, fluxes_error = {},{},{}
    for inst in instruments_lc:
        times_lc[inst], fluxes[inst], fluxes_error[inst] = np.loadtxt(datafolder + inst +'.lc.dat',
                                                                      unpack=True, usecols=(0,1,2))
        if (inst == 'TESSERACT+TESS') & (outlierIndices is not None):
            # mask outliers in TESS photometry
            times_lc[inst] = np.delete(times_lc[inst], outlierIndices)
            fluxes[inst] = np.delete(fluxes[inst], outlierIndices)
            fluxes_error[inst] = np.delete(fluxes_error[inst], outlierIndices)

            # include GPs for TESS`
            gp_times_lc[inst] = times_lc[inst]

        if plotPhot:
            # plot photometry
            plt.scatter(times_lc[inst], fluxes[inst],s=1)
            plt.title(inst)
            plt.show()

    return times_lc, fluxes, fluxes_error, gp_times_lc

def read_rv(datafolder, kms=True, subtract_median=True):
    """read RVs

    kms : boolean
        set true if data given in km/s. We'll convert it to m/s.
    subtract_median : boolean
        subtract median RV from RVs
    """
    times_rv, rvs, rvs_error = {},{},{}
    for inst in instruments_rv:
        times_rv[inst], rvs[inst], rvs_error[inst] = np.loadtxt(datafolder+inst+'.rv.dat',
                                                                unpack=True, usecols=(0,1,2))
        if kms:
            # convert km/s to m/s
            for data in [rvs[inst], rvs_error[inst]]:
                data *= 1000
        if subtract_median:
            # subtract median rvs
            rvs[inst] -= np.median(rvs[inst])
    return times_rv, rvs, rvs_error

def equilibriumTemp():
    """ Compute additional quantities
    """
    # current best-fit params
    a = 0.1122
    e = 0.575
    L = 1.196

    """ for instantaneous thermal equilibrium:
    use the approximation in Kaltenegger+2011 (equation 3), 
    assuming albedo=0, emissivity=1). Here is the orbital evolution for beta=0.5
    (i.e., tidally locked) and beta=1.0 (fast rotator). 
    Note that true anomaly is a spatial quantity and very different to the
    temporal evolution.
    """
    fig, ax = plt.subplots(figsize=plotstyle.set_size())
    for beta in [.5, 1.]:
        fig, ax = plots.plot_Teq_theta(a, e, L, fig=fig, ax=ax, albedo=0.,
                                       emissivity=1., beta=beta,
                                       label='{:.2f}'.format(beta))
    plt.legend(title='beta', loc='lower right')
    return fig, ax

def main(datafolder, out_folder, GP):
    """ run the fits and save the results to the `out_folder`"""

    priors, params = get_priors(GP)
    times_lc, fluxes, fluxes_error, gp_times_lc = read_photometry(datafolder,
                                                    plotPhot=False, outlierIndices=outlierIndices)
    times_rv, rvs, rvs_error = read_rv(datafolder, kms=True, subtract_median=True)

    if GP:
        GP_regressors = gp_times_lc
    else:
        GP_regressors = None

    dataset = juliet.load(
        priors=priors,
        t_lc=times_lc, y_lc=fluxes, yerr_lc=fluxes_error,
        t_rv=times_rv, y_rv=rvs, yerr_rv=rvs_error,
        GP_regressors_lc=GP_regressors,
        out_folder=out_folder, verbose=True)

    results = dataset.fit(use_dynesty=False, n_live_points=1500, ecclim=0.7,
                          dynamic=True,
                          pl=pl, pu=pu)

    lnZstr = r'Log - evidence: {0: .2f} $\pm$ {1: .2f}'.format(results.posteriors['lnZ'],\
                                                           results.posteriors['lnZerr'])
    print(lnZstr)
    with open(out_folder + "/lnZ={:.2f}.txt".format(results.posteriors['lnZ']), "w") as text_file:
        print(lnZstr, file=text_file)

    return results

def showResults(datafolder, out_folder, **fitKwargs):
    dataset = juliet.load(input_folder=out_folder)
    try:
        results = pickle.load(open(out_folder + '/results.pkl', 'rb'))
    except FileNotFoundError:
        results = dataset.fit(use_dynesty=False, dynamic=True,
                          **fitKwargs) # has to be ~same call as during fit

    # plot posteriors
    fig = plots.plot_cornerPlot(results, pl=results.pl, pu=results.pu,
                                quantiles=[0.16, 0.5, 0.84], show_titles=True,
                                title_kwargs={"fontsize": 16}, title_fmt='.2f',
                                rasterized=True,
                                label_kwargs={"fontsize": 16 })
    fig.savefig(out_folder + '/cornerPosteriors.pdf')

    # plot single posterior plots
    plots.plot_posteriors(results, out_folder)

    # Plot the photometry with fit
    if dataset.ninstruments_lc > 0:
        fig, axs = plots.plot_photometry(dataset, results)
        axs[0].legend(loc='lower left', ncol=99, bbox_to_anchor=(0., 1.),
                      frameon=False, columnspacing=1.6)
        fig.savefig(out_folder + '/photometryFitted.pdf')

        phasedPlots = plots.plot_phasedPhotometry(dataset, results, instrument=None)
        for inst in phasedPlots:
            phasedPlots[inst][0].savefig(out_folder + '/phasedPhot_{}.pdf'.format(inst))

    # plot RVs with fit
    if dataset.ninstruments_rv > 0:
        fig, ax = plots.plot_rv_fit(dataset, results)
        fig.savefig(out_folder + '/RVsFitted.pdf')



    # plot periodograms
    fig, axs = plots.plot_periodograms(datafolder+'TIC237913194_activity.dat',
                                       out_folder, results)

    # plot RV-BS
    fig, ax = plots.plot_RV_BS(datafolder+'TIC237913194_activity.dat', out_folder, results)

    lnZstr = r'Log - evidence: {0: .2f} $\pm$ {1: .2f}'.format(results.posteriors['lnZ'],\
                                                           results.posteriors['lnZerr'])
    print(lnZstr)
    return results


def printTables(out_folder):
    """ print Latex-formatted tables for priors, posteriors to files"""
    dataset = juliet.load(input_folder=out_folder)
    try:
        results = pickle.load(open(out_folder + '/results.pkl', 'rb'))
    except FileNotFoundError:
        results = dataset.fit(use_dynesty=False, dynamic=True,
                          **fitKwargs) # has to be ~same call as during fit
    # latextable.print_prior_table(dataset)
    latextable.print_posterior_table(dataset, results, precision=2)


if __name__ == "__main__":
    results = main(datafolder, out_folder, GP)
    pickle.dump(results, open(out_folder + '/results.pkl', 'wb'))

    # results = showResults(datafolder, out_folder, pl=pl, pu=pu)
    # printTables(out_folder)
