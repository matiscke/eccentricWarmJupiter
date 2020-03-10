import matplotlib.pyplot as plt
import corner
import juliet
import numpy as np
import aux
import os

try:
    from popsyntools import plotstyle
except ModuleNotFoundError:
    print('module "popsyntools" not found. Skipping plot styles therein.')

def plot_posteriors(julietResults, out_folder):
    # num_samps = len(julietResults.keys())
    if not os.path.exists(out_folder+'/posteriorplots/'):
        os.mkdir(out_folder+'/posteriorplots/')

    # exclude fixed parameters
    try:
        posteriors = julietResults.posteriors
    except AttributeError:
        # sometimes, juliet puts results into a tuple
        posteriors = julietResults[0].posteriors

    for k in posteriors['posterior_samples'].keys():
        if k != 'unnamed':
            val,valup,valdown = juliet.utils.get_quantiles(posteriors['posterior_samples'][k])
            print(k,':',val,' + ',valup-val,' - ',val-valdown)
            fig = plt.figure(figsize=(10,7))

            plt.hist(posteriors['posterior_samples'][k],
                     bins=int(len(posteriors['posterior_samples'][k])/50),
                     histtype='step')
            plt.axvline(x=val,color='cornflowerblue',lw=1.5,ls='--',
                            label='{} = {:.5}'.format(k, val))
            plt.axvline(x=valdown,color='cornflowerblue',lw=.5,ls='--')
            plt.axvline(x=valup,color='cornflowerblue',lw=.5,ls='--')
            plt.title('Posterior of : {}'.format(k))
            plt.xlabel(k)
            plt.ylabel('Frequency')
            plt.legend(loc=1)
            if k == 'P_p1':
              k = 'Period_p1'
              fil2save = out_folder+'/posteriorplots/Period_p1.pdf'
            else:
              fil2save = out_folder+'/posteriorplots/'+k+'.pdf'
            plt.tight_layout()
            fig.savefig(fil2save,dpi=400)
            # plt.show()
            plt.close(fig)


def plot_cornerPlot(julietResults, params, posterior_names=None, **kwargs):
    """ Produce a corner plot of posteriors from a juliet fit.

    Parameters
    ------------
    julietResults : results object
        a results object returned by juliet.fit()
    params : dictionary
        dict containing:
            keys: names of parameters included in the fit
            values: list, where 0th element is a string describing the distribution, and
            the 1st element are the hyperparameter(s) for this distribution.
        Example: {'r1_p1' : ['uniform', [0.,1]]}
    posterior_names : list, optional
        labels for the plot. If None, use keys of the params dictionary

    Returns
    --------
    fig : matplotlib figure
        figure containing the plot
    """
    # paramNames = list(params.keys())
    # if posterior_names is None:
    #     posterior_names = paramNames

    # exclude fixed parameters
    try:
        posteriors = [(name, julietResults.posteriors['posterior_samples'][name])
                  for name in params.keys() if params[name][0] != 'fixed']
    except AttributeError:
        # sometimes, juliet puts results into a tuple
        posteriors = [(name, julietResults[0].posteriors['posterior_samples'][name])
                      for name in params.keys() if params[name][0] != 'fixed']


    posterior_data = np.array([p[1] for p in posteriors]).T
    fig = corner.corner(posterior_data, #posterior_names,
                        labels=[aux.format(p[0]) for p in posteriors],
                        **kwargs)
    return fig

def plot_photometry(dataset, results):
    """ plot photometry and best fit from transit model.

    Parameters
    ------------
    dataset : dataset object
        dataset as returned by juliet.load()
    results : results object
        a results object returned by juliet.fit()

    Returns
    --------
    ax : matplotlib axis object
        axis containing the plot
    fig : matplotlib figure
        figure containing the plot
    """
    if isinstance(results, tuple):
        # sometimes, juliet.fit returns a tuple
        results = results[0]

    fig, ax = plt.subplots()
    ax.errorbar(dataset.times_lc['TESSERACT+TESS'], dataset.data_lc['TESSERACT+TESS'],
                 yerr=dataset.errors_lc['TESSERACT+TESS'], fmt='.', alpha=0.1)
    ax.plot(dataset.times_lc['TESSERACT+TESS'], results.lc.evaluate('TESSERACT+TESS'))

    # Plot portion of the lightcurve, axes, etc.:
    # plt.xlim([1326,1332])
    # plt.ylim([0.999,1.001])
    ax.set_xlabel('Time (BJD - 2458669)')
    ax.set_ylabel('Relative flux')
    return fig, ax


def plot_rv_fit(dataset, results):
    """ plot RV time series and best-fit model."""
    if isinstance(results, tuple):
        # sometimes, juliet.fit returns a tuple
        results = results[0]

    min_time, max_time = np.min(dataset.times_rv['FEROS']) - 30, \
                         np.max(dataset.times_rv['FEROS']) + 30
    model_times = np.linspace(min_time, max_time, 1000)
    # Now evaluate the model in those times, and substract the systemic-velocity to
    # get the Keplerian signal:
    keplerian = results.rv.evaluate('FEROS', t=model_times) - \
                np.median(results.posteriors['posterior_samples']['mu_FEROS'])


    fig, ax = plt.subplots()
    # ax.errorbar(dataset.times_rv['FEROS'], dataset.data_rv['FEROS'],
    #              yerr=dataset.errors_rv['FEROS'], fmt='.', alpha=0.1)

    # Now plot the (systematic-velocity corrected) RVs:
    instruments = dataset.inames_rv
    colors = ['cornflowerblue', 'orangered']
    for i in range(len(instruments)):
        instrument = instruments[i]
        # Evaluate the median jitter for the instrument:
        jitter = np.median(results.posteriors['posterior_samples']['sigma_w_' + instrument])
        # Evaluate the median systemic-velocity:
        mu = np.median(results.posteriors['posterior_samples']['mu_' + instrument])
        # Plot original data with original errorbars:
        ax.errorbar(dataset.times_rv[instrument] - 2458669, dataset.data_rv[instrument] - mu, \
                     yerr=dataset.errors_rv[instrument], fmt='o', \
                     mec=colors[i], ecolor=colors[i], elinewidth=3, mfc='white', \
                     ms=7, label=instrument, zorder=10)

        # Plot original errorbars + jitter (added in quadrature):
        ax.errorbar(dataset.times_rv[instrument] - 2458669, dataset.data_rv[instrument] - mu, \
                     yerr=np.sqrt(dataset.errors_rv[instrument] ** 2 + jitter ** 2), fmt='o', \
                     mec=colors[i], ecolor=colors[i], mfc='white', label=instrument, \
                     alpha=0.5, zorder=5)

    # Plot Keplerian model:
    ax.plot(model_times - 2458669, keplerian, color='black', zorder=1)
    plt.title('1 Planet Fit | Log-evidence: {:.2f} $\pm$ {:.2f}'.format(results.posteriors['lnZ'], \
                                                                          results.posteriors['lnZerr']))
    ax.set_xlabel('Time (BJD - 2458669)')
    ax.set_ylabel('RV [km/s]')
    return fig, ax


def plot_Teq_theta(a, e, L, fig=None, ax=None, albedo=0., emissivity=1.,
                   beta=1., **kwargs):
    """plot equilibrium temperature as a function of true anomaly theta."""
    if ax is None:
        fig, ax = plt.subplots(figsize=plotstyle.set_size())
    theta = np.linspace(0., 2*np.pi, 200)
    ax.plot(theta, aux.Teq(L, aux.r_of_theta(theta, a, e), albedo, emissivity,
                           beta), **kwargs)
    ax.set_xlabel('true anomaly [rad]')
    ax.set_ylabel('equilibrium temperature [K]')
    return fig, ax
