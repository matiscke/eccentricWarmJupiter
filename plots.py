import matplotlib.pyplot as plt
import corner
import numpy as np
import aux


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
    paramNames = list(params.keys())
    if posterior_names is None:
        posterior_names = paramNames
    posterior_data = np.array([julietResults.posteriors['posterior_samples'][name]
                               for name in paramNames if params[name][0] != 'fixed']).T
    fig = corner.corner(posterior_data, labels=posterior_names, **kwargs)
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
    fig, ax = plt.subplots()
    ax.errorbar(dataset.times_lc['TESSERACT+TESS'], dataset.data_lc['TESSERACT+TESS'],
                 yerr=dataset.errors_lc['TESSERACT+TESS'], fmt='.', alpha=0.1)
    ax.plot(dataset.times_lc['TESSERACT+TESS'], results.lc.evaluate('TESSERACT+TESS'))

    # Plot portion of the lightcurve, axes, etc.:
    # plt.xlim([1326,1332])
    # plt.ylim([0.999,1.001])
    ax.set_xlabel('Time (BJD - 2457000)')
    ax.set_ylabel('Relative flux')
    return fig, ax

def plot_Teq_theta(a, e, Rstar, Teff, albedo=0., emissivity=1., beta=1.):
    """plot equilibrium temperature as a function of true anomaly theta."""
    fig, ax = plt.subplots()
    theta = np.linspace(0., 2*np.pi, 200)
    ax.plot(theta, aux.Teq(aux.r_of_theta(theta, a, e), Rstar, Teff))
    ax.set_xlabel('true anomaly')
    ax.set_ylabel('equilibrium temperature')
    return fig, ax
