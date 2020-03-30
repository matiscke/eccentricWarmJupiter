import matplotlib as mpl
import matplotlib.pyplot as plt
import corner
import juliet
import numpy as np
import pandas as pd
import aux
import os

try:
    from astropy.timeseries import LombScargle
except ModuleNotFoundError:
    # the timeserie module moved at astropy 3.1 -> 3.2
    from astropy.stats import LombScargle

# prevent mpl from needing an X server
mpl.use('Agg')

try:
    from popsyntools import plotstyle
except ModuleNotFoundError:
    print('module "popsyntools" not found. Skipping plot styles therein.')


def plot_posteriors(julietResults, out_folder):
    """ plot individual posterior plots."""

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


def plot_phasedPhotometry(dataset, results, instrument=None):
    """ plot phased photometry and best fit from transit model.

    Parameters
    ------------
    dataset : dataset object
        dataset as returned by juliet.load()
    results : results object
        a results object returned by juliet.fit()
    instrument : string (optional)
        name of the instrument. If not given, create a plot for each instrument.

    Returns
    --------
    plots : dictionary
        dictionary with instrument names as keys. Each entry contains a tuple of
        (fig, axs):
        axs : list of matplotlib axis objects
            axis containing the plot
        fig : matplotlib figure
            figure containing the plot
    """
    if isinstance(results, tuple):
        # sometimes, juliet.fit returns a tuple
        results = results[0]

    if instrument is not None:
        instruments = [instrument]
    elif dataset.inames_lc is not None:
        # make a plot for each photometric instrument. Ignore provided figure or axes.
        instruments = dataset.inames_lc
        axs = None
    else:
        # no photometric data in the dataset. Nothing to plot.
        return

    plots = {}

    for inst in instruments:
        fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [5, 2]})

        # # Plot:
        # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2]}, facecolor='w',
        #                                figsize=(10, 4))
        #
        # # gs = gridspec.GridSpec(2, 1, height_ratios=[2,1])
        # nsamples = 1000
        #
        # fil = dataset.out_folder + '/lc_model_{}.pkl'.format(inst)
        # if os.path.isfile(fil):
        #     model = pickle.load(open(fil, 'rb'))
        #     lc_model, lc_up68, lc_low68, components = model['lc_model'], model['lc_up68'], model['lc_low68'], model[
        #         'components']
        # else:
        #     model = {}
        #
        #     model['lc_model'], model['lc_up68'], model['lc_low68'], model[
        #         'components'] = lc_model, lc_up68, lc_low68, components = results.lc.evaluate(inst, \
        #                                                                                       parameter_values=
        #                                                                                       posteriors[
        #                                                                                           'posterior_samples'], \
        #                                                                                       t=dataset.times_lc[
        #                                                                                           inst], \
        #                                                                                       nsamples=nsamples, \
        #                                                                                       return_err=True, \
        #                                                                                       return_components=True, \
        #                                                                                       GPregressors=
        #                                                                                       dataset.times_lc[
        #                                                                                           inst])
        #     model['nsamples'] = nsamples
        #
        #     pickle.dump(model, open(fil, 'wb'))



        for i_transit in dataset.numbering_transiting_planets:
            # transit_model = results.lc.model[inst]['deterministic']
            try:
                gp_model = results.lc.model[inst]['GP']
            except:
                gp_model = np.zeros(len(dataset.data_lc[inst]))
            try:
                P = results.posteriors['posterior_samples']['P_p{}'.format(i_transit)]
            except KeyError:
                P = dataset.priors['P_p{}'.format(i_transit)]['hyperparameters']
            try:
                t0 = results.posteriors['posterior_samples']['t0_p{}'.format(i_transit)]
            except KeyError:
                t0 = dataset.priors['t0_p{}'.format(i_transit)]['hyperparameters']

            phases_lc = juliet.utils.get_phases(dataset.times_lc[inst], P, t0)
            idx = np.argsort(phases_lc)


            # # define denser time array (didn't work, Diana is on it)
            # t_eva = np.linspace(min(dataset.times_lc[inst]),
            #                     max(dataset.times_lc[inst]), num=500)



            c_model, c_components = results.lc.evaluate(inst, t=dataset.times_lc[inst], \
                                                        # all_samples=True, \
                                                        return_components=True, \
                                                        GPregressors=dataset.times_lc[inst])


            axs[0].errorbar(phases_lc,
                             dataset.data_lc[inst] - (c_model - c_components['p{}'.format(i_transit)]) - gp_model, \
                             yerr=dataset.errors_lc[inst], fmt='.', \
                             alpha=0.1)

            # Plot transit-only (divided by mflux) model:
            axs[0].plot(phases_lc[idx], c_components['p{}'.format(i_transit)][idx] - c_components['lm'][idx],
                     color='black', zorder=10)
            axs[1].axhline(y=0, ls='--', color='k', alpha=0.5)
            # axs[0].yaxis.set_major_formatter(plt.NullFormatter())
            try:
                axs[0].set_title('P = {:.5f} t0 = {:.5f}'.format(P, t0))
            except:
                axs[0].set_title('P = {:.5f} t0 = {:.5f}'.format(np.median(P), np.median(t0)))

            axs[0].set_ylabel('Relative flux')
            axs[0].set_xlim([-0.015, 0.015])  ### CHANGE THIS
            # axs[0].set_ylim([0.9985,1.0015]) ### CHANGE THIS
            axs[0].minorticks_on()
            axs[1].errorbar(phases_lc, (dataset.data_lc[inst] - c_model - gp_model) * 1e6, \
                         yerr=dataset.errors_lc[inst], fmt='.', alpha=0.1)
            axs[1].set_ylabel('Residuals (ppm)')
            axs[1].set_xlabel('Phases')
            plt.tight_layout()
            # fig.subplots_adjust(hspace=0)  # to make the space between rows smaller
            axs[1].minorticks_on()
        plots[inst] = (fig, axs)

    return plots


def plot_periodograms(activityFile, plot_dir, results):
    """
    Produce periodograms of RV and activity indices

    @author: Melissa Hobson
    adapted by Martin Schlecker

    Parameters
    ------------
    activityFile :  string
        file containing the reduced time series
    plot_dir : string
        directory for the created plots
    results : results object
        a results object returned by juliet.fit()

    Returns
    --------
    fig : matplotlib figure
        figure containing the plot
    axs : array
        contains axis objects with the plots
    """

    font = {'size'   : 15}
    mpl.rc('font', **font)
    bbox_props = {'boxstyle':"round",
         'fc':"w",
         'edgecolor':"w",
         # 'ec':0.5,
         'alpha':0.9
    }

    # =============================================================================
    # Read data in

    # FEROS data
    feros_dat = np.genfromtxt(activityFile, names=True)
    feros_dat = pd.DataFrame(feros_dat).replace(-999, np.nan)

    transit_per = np.median(results.posteriors['posterior_samples']['P_p1'])

    # =============================================================================
    # Periodograms - FEROS

    f_min = 1/(feros_dat['BJD_OUT'].max() - feros_dat['BJD_OUT'].min())
    f_max = None

    #RV
    # variables
    bjd_feros = feros_dat['BJD_OUT']
    RV_feros = feros_dat['RV']
    RV_E_feros = feros_dat['RV_E']

    # create periodogram
    rv_ls = LombScargle(bjd_feros, RV_feros, RV_E_feros)
    rv_frequency, rv_power = rv_ls.autopower(minimum_frequency=f_min,
                                             maximum_frequency=f_max)

    # Get FAP levels
    probabilities = [0.01, 0.005, 0.001]
    labels = ['1.00% FAP', '0.50% FAP', '0.01% FAP']
    ltype = ['solid', 'dashed', 'dotted']
    rv_faps = rv_ls.false_alarm_level(probabilities, method='bootstrap')

    # H alpha
    # variables
    ha_feros = feros_dat['HALPHA']
    ha_e_feros = feros_dat['HALPHA_E']

    # create periodogram
    ha_ls = LombScargle(bjd_feros, ha_feros, ha_e_feros)
    ha_frequency, ha_power = ha_ls.autopower(minimum_frequency=f_min,
                                             maximum_frequency=f_max)

    # Get FAP levels
    ha_faps = ha_ls.false_alarm_level(probabilities, method='bootstrap')

    # log Rhk
    # variables
    rhk_feros = feros_dat['LOG_RHK'].dropna()
    rhk_e_feros = feros_dat['LOGRHK_E'].dropna()
    bjd_rhk = bjd_feros.iloc[rhk_feros.index]


    # create periodogram
    rhk_ls = LombScargle(bjd_rhk, rhk_feros, rhk_e_feros)
    rhk_frequency, rhk_power = rhk_ls.autopower(minimum_frequency=f_min,
                                             maximum_frequency=f_max)

    # Get FAP levels
    rhk_faps = rhk_ls.false_alarm_level(probabilities, method='bootstrap')

    # Na II
    # variables
    na_feros = feros_dat['NA_II']
    na_e_feros = feros_dat['NA_II_E']

    # create periodogram
    na_ls = LombScargle(bjd_feros, na_feros, na_e_feros)
    na_frequency, na_power = na_ls.autopower(minimum_frequency=f_min,
                                             maximum_frequency=f_max)

    # Get FAP levels
    na_faps = na_ls.false_alarm_level(probabilities, method='bootstrap')

    # He I
    # variables
    he_feros = feros_dat['HE_I']
    he_e_feros = feros_dat['HE_I_E']

    # create periodogram
    he_ls = LombScargle(bjd_feros, he_feros, he_e_feros)
    he_frequency, he_power = he_ls.autopower(minimum_frequency=f_min,
                                             maximum_frequency=f_max)

    # Get FAP levels
    he_faps = he_ls.false_alarm_level(probabilities, method='bootstrap')

    # =============================================================================
    # Plot the data
    # figsize = plotstyle.set_size(subplot=[5,1]) # (11, 21)
    figsize = (8, 10)
    fig, axs = plt.subplots(5,1, figsize=figsize, sharex=True, sharey=True,
                            gridspec_kw = {'wspace':0, 'hspace':0.08})


    # # RV timeseries
    # axs[0].errorbar(bjd_feros, RV_feros, yerr=RV_E_feros, fmt='o')
    # axs[0].set_xlabel('BJD')
    # axs[0].set_ylabel('RV [km/s]')

    # RV periodogram
    annotOffsets = [-.12, 0, .12]
    axs[0].plot(rv_frequency, rv_power)
    for ind in range(len(rv_faps)):
        axs[0].axhline(rv_faps[ind], xmax=0.81,
            label = labels[ind], lw=1.5, linestyle = ltype[ind], c='black')
        axs[0].annotate(labels[ind], xy=[.815, rv_faps[ind]], va='center',
                        xytext=[.86, rv_faps[1] + annotOffsets[ind]], size=10,
                        xycoords=('axes fraction', 'data'), arrowprops=dict(arrowstyle="-"))
    axs[0].axvline(1/transit_per,  lw=1.5, linestyle='dashed', color='C1')

    # axs[0].set_xscale('log')
    # axs[0].set_xlabel('Frequency [1/d]')
    axs[0].set_ylabel('Power')
    axs[0].annotate('P = {:.2f} d'.format(transit_per), [1/transit_per, 1.05], color='C1',
                    ha='center', xycoords=('data', 'axes fraction'))
    axs[0].annotate('RV', xy=(0, 1.01), xytext=(.02, .84), size=15, bbox=bbox_props,
                               ha='left', va='center', xycoords='axes fraction', textcoords='axes fraction')


    # # Halpha timeseries
    # plt.subplot(4, 3, 2)
    # plt.errorbar(bjd_feros, ha_feros, yerr=ha_e_feros, fmt='o')
    # plt.set_xlabel('BJD')
    # plt.set_ylabel('H ALPHA')

    # Halpha periodogram
    axs[1].plot(ha_frequency, ha_power)
    for ind in range(len(ha_faps)):
        axs[1].axhline(ha_faps[ind],
            label = labels[ind], lw=1.5, linestyle = ltype[ind], c='black')
    axs[1].axvline(1/transit_per, lw=1.5, linestyle='dashed', color='C1')
    # axs[1].set_xscale('log')
    # axs[1].set_xlabel('Period [d]')
    axs[1].set_ylabel('Power')
    axs[1].annotate(r'H$_\alpha$', xy=(0, 1.01), xytext=(.02, .84), size=15, bbox=bbox_props,
                               ha='left', va='center', xycoords='axes fraction', textcoords='axes fraction')

    # # log RHK timeseries
    # plt.subplot(4, 3, 3)
    # plt.errorbar(bjd_feros, rhk_feros, yerr=rhk_e_feros, fmt='o')
    # plt.set_xlabel('BJD'HKk)
    # plt.set_ylabel('LOG RHK')
    # log Rhk periodogram
    axs[2].plot(rhk_frequency, rhk_power)
    for ind in range(len(rhk_faps)):
        axs[2].axhline(rhk_faps[ind],
            label = labels[ind], lw=1.5, linestyle = ltype[ind], c='black')
    axs[2].axvline(1/transit_per, lw=1.5, linestyle='dashed', color='C1')
    # axs[2].set_xscale('log')
    # axs[2].set_xlabel('Period [d]')
    axs[2].set_ylabel('Power')
    axs[2].annotate(r'log($R^\prime_{HK}$)', xy=(0, 1.01), xytext=(.02, .84), size=15, bbox=bbox_props,
                               ha='left', va='center', xycoords='axes fraction', textcoords='axes fraction')

    # # Na II timeseries
    # plt.subplot(4, 3, 8)
    # plt.errorbar(bjd_feros, na_feros, yerr=na_e_feros, fmt='o')
    # plt.set_xlabel('BJD')
    # plt.set_ylabel('NA II')
    # Na II periodogram
    axs[3].plot(na_frequency, na_power)
    for ind in range(len(na_faps)):
        axs[3].axhline(na_faps[ind],
            label = labels[ind], lw=1.5, linestyle = ltype[ind], c='black')
    axs[3].axvline(1/transit_per, lw=1.5, linestyle='dashed', color='C1')
    # axs[3].set_xscale('log')
    # axs[3].set_xlabel('Period [d]')
    axs[3].set_ylabel('Power')
    axs[3].annotate(r'Na II', xy=(0, 1.01), xytext=(.02, .84), size=15, bbox=bbox_props,
                               ha='left', va='center', xycoords='axes fraction', textcoords='axes fraction')

    # # HeI timeseries
    # plt.subplot(4, 3, 9)
    # plt.errorbar(bjd_feros, he_feros, yerr=he_e_feros, fmt='o')
    # plt.set_xlabel('BJD')
    # plt.set_ylabel('HE I')
    # HeI periodogram
    axs[4].plot(he_frequency, he_power)
    for ind in range(len(he_faps)):
        axs[4].axhline(he_faps[ind],
            label = labels[ind], lw=1.5, linestyle = ltype[ind], c='black')
    axs[4].axvline(1/transit_per, lw=1.5, linestyle='dashed', color='C1')
    # axs[4].set_xscale('log')
    # axs[4].set_xlabel('Period [d]')
    axs[4].set_xlabel('Frequency [1/d]')
    axs[4].set_ylabel('Power')
    axs[4].annotate(r'He I', xy=(0, 1.01), xytext=(.02, .84), size=15, bbox=bbox_props,
                               ha='left', va='center', xycoords='axes fraction', textcoords='axes fraction')

    # some eye candy
    [ax.set_xlim([0.005, 0.3]) for ax in axs]
    # [ax.set_ylim([0,.9]) for ax in axs]
    [ax.tick_params(direction='in', top=True, right=True) for ax in axs]
    # fig.subplots_adjust(hspace = .03, wspace=0.4)
    plt.show()
    fig.savefig(plot_dir + '/periodograms.pdf')

    return fig, axs
