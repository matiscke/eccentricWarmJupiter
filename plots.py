import matplotlib
import matplotlib.pyplot as plt
import corner
import juliet
import numpy as np
import aux
import os

# prevent mpl from needing an X server
matplotlib.use('Agg')

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
        # sometimes, juliet puts julietResults into a tuple
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


def plot_cornerPlot(julietResults, posterior_names=None, pl=0., pu=1., **kwargs):
    """ Produce a corner plot of posteriors from a juliet fit.

    Parameters
    ------------
    julietResults : results object
        a results object returned by juliet.fit()
    posterior_names : list, optional
        labels for the plot. If None, use keys of the params dictionary

    Returns
    --------
    fig : matplotlib figure
        figure containing the plot

    Notes
    ------
    Assumes quadratic limb darkening for an instrument 'TESSERACT+TESS' and
    linear limb darkening for 'CHAT+i' (only when present in julietResults object)

    """
    # paramNames = list(params.keys())
    # if posterior_names is None:
    #     posterior_names = paramNames

    # back-transform r1, r2 to b, p and q1, q2 to u1, u2
    if 'r1_p1' in julietResults.posteriors['posterior_samples']:
        r1, r2 = julietResults.posteriors['posterior_samples']['r1_p1'], \
             julietResults.posteriors['posterior_samples']['r2_p1']
        b, p = juliet.utils.reverse_bp(r1, r2, pl, pu)
    else:
        b, p = None, None

    q1_tess, q2_tess = julietResults.posteriors['posterior_samples']['q1_TESSERACT+TESS'], \
                       julietResults.posteriors['posterior_samples']['q2_TESSERACT+TESS']
    u1_tess, u2_tess = juliet.utils.reverse_ld_coeffs('quadratic', q1_tess, q2_tess)
    try:
        q1_chat = julietResults.posteriors['posterior_samples']['q1_CHAT+i']
        u1_chat, u1_chat = juliet.utils.reverse_ld_coeffs('linear', q1_chat, q1_chat)
    except:
        pass

    # back-transfrom ecc, omega parametrization
    secosomega = julietResults.posteriors['posterior_samples']['secosomega_p1']
    sesinomega = julietResults.posteriors['posterior_samples']['sesinomega_p1']
    ecc = secosomega ** 2 + sesinomega ** 2
    omega = np.arccos(secosomega / np.sqrt(ecc)) * np.pi/180


    # extract posteriors, excluding fixed parameters
    try:
        posteriorSamples = julietResults.posteriors['posterior_samples']
    except AttributeError:
        posteriorSamples = julietResults[0].posteriors['posterior_samples']

    # shift to relative t0
    posteriorSamples['t0_p1'] -= 2458000

    posteriors = []
    for name in julietResults.data.priors:
        if (name not in ['r1_p1','r2_p1','q1_TESSERACT+TESS','q2_TESSERACT+TESS',
                'q1_CHAT+i', 'secosomega_p1', 'sesinomega_p1']) & \
                (julietResults.data.priors[name]['distribution'] != 'fixed'):
            # consider all non-fixed params, except special parametrizations
            if julietResults.data.priors[name]['distribution'] == 'loguniform':
                # plot log. distributed params in log
                posteriors.append(('log '+name, np.log10(posteriorSamples[name])))
            else:
                posteriors.append((name,posteriorSamples[name]))

    # include special parametrizations
    if b is not None:
        posteriors.append(('b', b))
        posteriors.append(('p', p))
    posteriors.append(('u1_TESSERACT+TESS', u1_tess))
    posteriors.append(('u2_TESSERACT+TESS', u2_tess))
    posteriors.append(('ecc', ecc))
    posteriors.append(('omega', omega))
    try:
        posteriors.append(('u1_CHAT+i', u1_chat))
    except:
        pass

    posterior_data = np.array([p[1] for p in posteriors]).T
    fig = corner.corner(posterior_data, #posterior_names,
                        labels=[aux.format(p[0]) + '\n' for p in posteriors],
                        **kwargs)
    # tune look of corner figure
    caxes = fig.axes
    for ax in caxes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_label_coords(0.5, -0.45)
        ax.yaxis.set_label_coords(-0.35, 0.5)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.09, top=0.97,
                        wspace=.15, hspace=.15)
    return fig


def plot_photometry(dataset, results, fig=None, axs=None, instrument='TESSERACT+TESS'):
    """ plot photometry and best fit from transit model.

    Parameters
    ------------
    dataset : dataset object
        dataset as returned by juliet.load()
    results : results object
        a results object returned by juliet.fit()
    fig : matplotlib figure object, optional
        figure to plot on
    axs : list (opional)
        list containing axis objects
    instrument : string (optional)
        name of the instrument

    Returns
    --------
    axs : list of matplotlib axis objects
        axis containing the plot
    fig : matplotlib figure
        figure containing the plot
    """
    if isinstance(results, tuple):
        # sometimes, juliet.fit returns a tuple
        results = results[0]

    transit_model, transit_up68, transit_low68 = results.lc.evaluate(instrument,
                                                                     return_err=True)
    transit_model, transit_up95, transit_low95 = results.lc.evaluate(instrument,
                                                                     return_err=True, alpha=.9545)

    if axs is None:
        fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [5, 2]})
    axs[0].errorbar(dataset.times_lc[instrument]- 2458000, dataset.data_lc[instrument],
                 yerr=dataset.errors_lc[instrument], fmt = '.', alpha=.66,
                 elinewidth = .5, ms = 1, color='black', label = 'TESS')
    axs[0].plot(dataset.times_lc[instrument]- 2458000, transit_model,
                lw=0.5, label='Full model')
    axs[0].fill_between(dataset.times_lc[instrument]- 2458000, transit_up68, transit_low68,
                    color='cornflowerblue', alpha=0.5, zorder=5)
    # axs[0].fill_between(dataset.times_lc[instrument]- 2458000, transit_up95, transit_low95,
    #                 color='cornflowerblue', alpha=0.3, zorder=6)

    # Now the residuals:
    axs[1].errorbar(dataset.times_lc[instrument] - 2458000,
                    (dataset.data_lc[instrument] - transit_model) * 1e6,
                 dataset.errors_lc[instrument] * 1e6, fmt='.', alpha=0.66, elinewidth=.5,
                 ms=1, color='black', label='residuals')

    axs[1].set_ylabel('Residuals (ppm)')
    axs[1].set_xlabel('Time (BJD - 2458000)')
    axs[1].set_xlim(np.min(dataset.times_lc[instrument] - 2458000), np.max(dataset.times_lc[instrument] - 2458000))

    # Plot portion of the lightcurve, axes, etc.:
    # plt.xlim([1326,1332])
    # plt.ylim([0.999,1.001])
    axs[1].set_xlabel('Time (BJD - 2458000)')
    [ax.set_ylabel('Relative flux') for ax in axs]
    return fig, axs


def plot_rv_fit(dataset, results):
    """ plot RV time series and best-fit model."""
    if isinstance(results, tuple):
        # sometimes, juliet.fit returns a tuple
        results = results[0]

    min_time, max_time = np.min(dataset.times_rv['FEROS']) - 10, \
                         np.max(dataset.times_rv['FEROS']) + 10
    model_times = np.linspace(min_time, max_time, 1000)
    # Now evaluate the model in those times, including 1 and 2 sigma CIs,
    # and substract the systemic-velocity to get the Keplerian signal:
    keplerian, up68, low68 = results.rv.evaluate('FEROS', t=model_times,
                                                 return_err=True) - \
                np.median(results.posteriors['posterior_samples']['mu_FEROS'])
    keplerian, up95, low95 = results.rv.evaluate('FEROS', t=model_times,
                                                 return_err=True, alpha=.9545) - \
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
        ax.errorbar(dataset.times_rv[instrument] - 2458000, dataset.data_rv[instrument] - mu, \
                     yerr=dataset.errors_rv[instrument], fmt='o', \
                     mec=colors[i], ecolor=colors[i], elinewidth=3, mfc='white', \
                     ms=7, label=instrument, zorder=10)

        # Plot original errorbars + jitter (added in quadrature):
        ax.errorbar(dataset.times_rv[instrument] - 2458000, dataset.data_rv[instrument] - mu, \
                     yerr=np.sqrt(dataset.errors_rv[instrument] ** 2 + jitter ** 2), fmt='o', \
                     mec=colors[i], ecolor=colors[i], mfc='white', label=instrument, \
                     alpha=0.5, zorder=5)

    # Plot Keplerian model and CIs:
    ax.plot(model_times - 2458000, keplerian, color='black', zorder=1)
    ax.fill_between(model_times - 2458000, up68, low68, \
                     color='cornflowerblue', alpha=0.5, zorder=5)
    ax.fill_between(model_times - 2458000, up95, low95, \
                    color='cornflowerblue', alpha=0.3, zorder=6)

    plt.title('Log-evidence: {:.2f} $\pm$ {:.2f}'.format(results.posteriors['lnZ'], \
                                                                          results.posteriors['lnZerr']))
    ax.set_xlabel('Time (BJD - 2458000)')
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
