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

# style matplotlib plots
try:
    from popsyntools import plotstyle
except ModuleNotFoundError:
    print('module "popsyntools" not found. Skipping plot styles therein.')

figure = {'dpi' : 200,
          'subplot.left'    : 0.16,   # the left side of the subplots of the figure
          'subplot.bottom'  : 0.21,   # the bottom of the subplots of the figure
          'subplot.right'   : 0.98,   # the right side of the subplots of the figure
          'subplot.top'     : 0.97,   # the top of the subplots of the figure
          'subplot.hspace'  : 0.15,    # height reserved for space between subplots
          'figsize' : [4.3, 3.2]}
mpl.rc('figure', **figure)
mpl.rc('savefig', bbox = 'tight', dpi = 200)

colors_rv = ['orangered', 'cornflowerblue']


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


def plot_cornerPlot(julietResults, posterior_names=None, pl=0., pu=1., reverse=False, fig=None, axes=None, **kwargs):
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

    if 'q1_TESSERACT+TESS' in julietResults.posteriors['posterior_samples']:
        q1_tess, q2_tess = julietResults.posteriors['posterior_samples']['q1_TESSERACT+TESS'], \
                           julietResults.posteriors['posterior_samples']['q2_TESSERACT+TESS']
        u1_tess, u2_tess = juliet.utils.reverse_ld_coeffs('quadratic', q1_tess, q2_tess)
    else:
        u1_tess = None
        u2_tess = None
    if 'q1_CHAT+i' in julietResults.posteriors['posterior_samples']:
        q1_chat = julietResults.posteriors['posterior_samples']['q1_CHAT+i']
        u1_chat, u1_chat = juliet.utils.reverse_ld_coeffs('linear', q1_chat, q1_chat)
    else:
        u1_chat = None
        u2_chat = None

    if 'secosomega_p1' in julietResults.posteriors['posterior_samples']:
        # back-transfrom ecc, omega parametrization
        secosomega = julietResults.posteriors['posterior_samples']['secosomega_p1']
        sesinomega = julietResults.posteriors['posterior_samples']['sesinomega_p1']
        ecc = secosomega ** 2 + sesinomega ** 2
        omega = np.arccos(secosomega / np.sqrt(ecc)) * 180/np.pi
    else:
        ecc = None
        omega = None


    # extract posteriors, excluding fixed parameters
    try:
        posteriorSamples = julietResults.posteriors['posterior_samples'].copy()
    except AttributeError:
        posteriorSamples = julietResults[0].posteriors['posterior_samples'].copy()

    posteriors = []
    for name in julietResults.data.priors:
        if (name not in ['r1_p1','r2_p1','q1_TESSERACT+TESS','q2_TESSERACT+TESS',
                         'sigma_w_TESSERACT+TESS',
                'q1_CHAT+i', 'secosomega_p1', 'sesinomega_p1']) & \
                (julietResults.data.priors[name]['distribution'] != 'fixed'):
            # consider all non-fixed params, except special parametrizations
            if julietResults.data.priors[name]['distribution'] == 'loguniform':
                # plot log. distributed params in log
                posteriors.append(('log '+name, np.log10(posteriorSamples[name])))
            else:
                posteriors.append((name,posteriorSamples[name]))

        if (name in ['sigma_w_TESSERACT+TESS']) & \
                (julietResults.data.priors[name]['distribution'] != 'fixed'):
            # dirty hack for some params that shouldn't be plotted in log
            posteriors.append((name, posteriorSamples[name]))

    # include special parametrizations
    if ecc is not None:
        posteriors.append(('ecc', ecc))
        posteriors.append(('omega', omega))
    if b is not None:
        posteriors.append(('b', b))
        posteriors.append(('p', p))
    if u1_tess is not None:
        posteriors.append(('u1_TESSERACT+TESS', u1_tess))
        posteriors.append(('u2_TESSERACT+TESS', u2_tess))
    if u1_chat is not None:
        posteriors.append(('u1_CHAT+i', u1_chat))

    # # select parameters for the plot
    if posterior_names is not None:
        posterior_subset = []

        # for label, posterior_samples in zip([p[0] for p in posteriors], posteriors):  # [p[1] for p in posteriors]):
        #     if label in posterior_names:
        #         posterior_subset.append(posterior_samples)

        for label in posterior_names:
            posterior_subset.append([label, julietResults.posteriors['posterior_samples'][label]])

    else:
        posterior_subset = posteriors

    posterior_data = np.array([p[1] for p in posterior_subset]).T
    fig = corner.corner(posterior_data, fig=fig, axes=axes, #posterior_names,
                        labels=[aux.format(p[0]) + '\n' for p in posterior_subset], reverse=reverse,
                        **kwargs)
    # tune look of corner figure
    if axes is None:
        axes = fig.axes
    else:
        axes = axes.flatten()
    if not reverse:
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.xaxis.set_label_coords(0.5, -.6)
            ax.yaxis.set_label_coords(-0.35, 0.5)
    else:
        for ax in axes:
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.xaxis.set_label_coords(0.5, -0.2)
            ax.yaxis.set_label_coords(-0.35, 0.5)
        fig.subplots_adjust(left=0.08, right=0.995, bottom=0.09, top=0.97,
                        wspace=.15, hspace=.15)
    return fig


def plot_photometry(dataset, results, fig=None, axs=None, instrument=None):
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

    if instrument is not None:
        instruments = [instrument]
    elif dataset.inames_lc is not None:
        # make a plot for each photometric instrument. Ignore provided figure or axes.
        instruments = dataset.inames_lc
        axs = None
    else:
        # no photometric data in the dataset. Nothing to plot.
        return

    transit_model, transit_up68, transit_low68 = results.lc.evaluate(instrument,
                                                                     return_err=True)
    transit_model, transit_up95, transit_low95 = results.lc.evaluate(instrument,
                                                                     return_err=True, alpha=.9545)

    if axs is None:
        fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [5, 2]},
                                figsize = [8.6, 3.2])
    axs[0].errorbar(dataset.times_lc[instrument]- 2458000, dataset.data_lc[instrument],
                 yerr=dataset.errors_lc[instrument], **aux.photPlotParams(), label = aux.label(instrument))
    axs[0].plot(dataset.times_lc[instrument]- 2458000, transit_model,
                lw=1, label='model')
    axs[0].fill_between(dataset.times_lc[instrument]- 2458000, transit_up68, transit_low68,
                    color='cornflowerblue', alpha=0.6, zorder=5)
    axs[0].fill_between(dataset.times_lc[instrument]- 2458000, transit_up95, transit_low95,
                    color='cornflowerblue', alpha=0.2, zorder=5)

    # Now the residuals:
    axs[1].errorbar(dataset.times_lc[instrument] - 2458000,
                    (dataset.data_lc[instrument] - transit_model) * 1e6,
                 dataset.errors_lc[instrument] * 1e6, **aux.photPlotParams(), label='residuals')
    axs[1].axhline(0, ls='--', lw=1, color='k', alpha=0.5)

    axs[1].set_ylabel('residuals [ppm]')
    axs[1].set_xlabel('Time [BJD - 2458000]')
    axs[1].set_xlim(np.min(dataset.times_lc[instrument] - 2458000), np.max(dataset.times_lc[instrument] - 2458000))

    # Plot portion of the lightcurve, axes, etc.:
    # plt.xlim([1326,1332])
    # plt.ylim([0.999,1.001])
    axs[1].set_xlabel('Time [BJD - 2458000]')
    axs[0].set_ylabel('relative flux')
    axs[1].set_ylabel('residuals [ppm]')

    leg = axs[0].legend(loc='lower left', ncol=99, bbox_to_anchor=(0., 1.),
                        frameon=False, columnspacing=1.6)
    return fig, axs


def plot_phasedPhotometry(dataset, results, instrument=None, color='C0'):
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

    times_lc = results.data.times_lc
    data_lc = results.data.data_lc
    errors_lc = results.data.errors_lc

    numbering_planets_transit = results.data.numbering_transiting_planets
    instruments_lc = results.data.inames_lc
    for inst in instruments:

        try:
            _ = results.lc.evaluate(inst, t = times_lc[inst], \
                )
            gp_data_model = np.zeros(len(times_lc[inst]))
        except:
            _ = results.lc.evaluate(inst, t = times_lc[inst], \
                GPregressors=times_lc[inst],\
                )
            gp_data_model = results.lc.model[inst]['GP']
        det_data_model = results.lc.model[inst]['deterministic']

        for i_transit in numbering_planets_transit:
            try:
                P = np.median(results.posteriors['posterior_samples']['P_p{}'.format(i_transit)])
            except KeyError:
                P = results.data.priors['P_p{}'.format(i_transit)]['hyperparameters']
            try:
                t0 = np.median(results.posteriors['posterior_samples']['t0_p{}'.format(i_transit)])
            except KeyError:
                t0 = results.data.priors['t0_p{}'.format(i_transit)]['hyperparameters']

            fig,axs = plt.subplots(2,1,sharex=True, gridspec_kw = {'height_ratios':[5,2]})
            phases_lc = juliet.utils.get_phases(times_lc[inst], P, t0)

            model_phases = np.linspace(-0.04,0.04,1000)
            model_times = model_phases*P + t0

            try:
                model_lc, transit_up68, transit_low68, model_components = results.lc.evaluate(inst, t = model_times, \
                                        return_components = True,\
                                        return_err=True, alpha=0.68)
                _, transit_up95, transit_low95, _ = results.lc.evaluate(inst, t = model_times, \
                                        return_components = True,\
                                        return_err=True, alpha=0.95)
                _, transit_up99, transit_low99, _ = results.lc.evaluate(inst, t = model_times, \
                                        return_components = True,\
                                        return_err=True, alpha=0.99)
            except:
                model_lc, transit_up68, transit_low68, model_components = results.lc.evaluate(inst, t = model_times, \
                                        return_components = True,\
                                        GPregressors=model_times, \
                                        return_err=True, alpha=0.68)
                _, transit_up95, transit_low95, _ = results.lc.evaluate(inst, t = model_times, \
                                        return_components = True,\
                                        GPregressors=model_times, \
                                        return_err=True, alpha=0.95)
                _, transit_up99, transit_low99, _ = results.lc.evaluate(inst, t = model_times, \
                                        return_components = True,\
                                        GPregressors=model_times, \
                                        return_err=True, alpha=0.99)

            axs[0].errorbar(phases_lc, data_lc[inst] - gp_data_model, \
                 yerr = errors_lc[inst], **aux.photPlotParams(), label=aux.label(inst),
                            zorder=6)
            axs[0].plot(model_phases, model_lc, lw=1, color='black', zorder=7)

            axs[0].fill_between(model_phases,transit_up68,transit_low68,\
                     color='cornflowerblue', alpha=0.6,zorder=5, label='model')
            axs[0].fill_between(model_phases,transit_up95,transit_low95,\
                     color='cornflowerblue',alpha=0.3,zorder=5)
            # axs[0].fill_between(model_phases,transit_up99,transit_low99,\
            #          color='cornflowerblue',alpha=0.2,zorder=5)

            axs[1].errorbar(phases_lc, (data_lc[inst]-det_data_model-gp_data_model)*1e6, \
                         yerr = errors_lc[inst]*1e6, **aux.photPlotParams())
            axs[1].axhline(y=0, ls='--', lw=1, color='k', alpha=0.5)
            # ax2.yaxis.set_major_formatter(plt.NullFormatter())
            # try:
            #     axs[0].set_title('P = {:.5f} t0 = {:.5f}'.format(P, t0))
            # except:
            #     axs[0].set_title('P = {:.5f} t0 = {:.5f}'.format(np.median(P), np.median(t0)))


            # ax2.set_ylim([0.9985,1.0015]) ### CHANGE THIS
            axs[0].minorticks_on()
            axs[0].set_ylabel('relative flux')
            axs[1].set_ylabel('residuals [ppm]')
            axs[1].set_xlabel('orbital phase')
            leg = axs[0].legend(loc='lower left', ncol=99, bbox_to_anchor=(0., 1.),
                                frameon=False, columnspacing=1.6)
            axs[1].minorticks_on()
            # axs[0].yaxis.set_tick_params(labelsize=fontsize_phot_ticks)
            # axs[1].xaxis.set_tick_params(labelsize=fontsize_phot_ticks)
            # axs[1].yaxis.set_tick_params(labelsize=fontsize_phot_ticks)

            # custom x limits, adapt for specific case
            if inst == 'CHAT+i':
                plt.xlim([-0.007,0.007])
                axs[1].set_ylim([-5200, 5200])
            elif inst == 'TESSERACT+TESS':
                axs[0].set_xlim([-0.015,0.015])
                axs[1].set_ylim([-2500, 2500])
            elif inst == 'LCOGT':
                axs[0].set_xlim([-0.004, 0.004])
                axs[1].set_ylim([-2500, 2500])
            else:
                axs[0].set_xlim([-0.03,0.03])

            # plt.tight_layout()
            # fig.subplots_adjust(hspace=0) # to make the space between rows smaller
            # plt.savefig(resultsPath+'/phased_lc_{}_pl{}.pdf'.format(inst,i_transit), dpi=700)
        plots[inst] = (fig, axs)
    return plots


def plot_rv_fit(dataset, results):
    """ plot RV time series and best-fit model.
    """
    if isinstance(results, tuple):
        # sometimes, juliet.fit returns a tuple
        results = results[0]

    min_time, max_time = np.min(dataset.times_rv['FEROS']) - 10, \
                         np.max(dataset.times_rv['FEROS']) + 10
    model_times = np.linspace(min_time, max_time, 1000)
    # Now evaluate the model in those times, including 1 and 2 sigma CIs,
    # and substract the systemic-velocity to get the Keplerian signal:
    keplerian, up68, low68 = results.rv.evaluate('FEROS', t=model_times,
                                                 return_err=True, all_samples = True) - \
                np.median(results.posteriors['posterior_samples']['mu_FEROS'])
    keplerian, up95, low95 = results.rv.evaluate('FEROS', t=model_times,
                                                 return_err=True, all_samples = True, alpha=.9545) - \
                np.median(results.posteriors['posterior_samples']['mu_FEROS'])


    fig, axs = plt.subplots(2, sharex=True, figsize=[8.6, 3.2],
                        gridspec_kw={'height_ratios': [5, 2]})
    # axs[0].errorbar(dataset.times_rv['FEROS'], dataset.data_rv['FEROS'],
    #              yerr=dataset.errors_rv['FEROS'], fmt='.', alpha=0.1)

    # Now plot the (systematic-velocity corrected) RVs:
    instruments = dataset.inames_rv
    colors = colors_rv
    for i in range(len(instruments)):
        instrument = instruments[i]
        # Evaluate the median jitter for the instrument:
        jitter = np.median(results.posteriors['posterior_samples']['sigma_w_' + instrument])
        # Evaluate the median systemic-velocity:
        mu = np.median(results.posteriors['posterior_samples']['mu_' + instrument])
        # Plot original data with original errorbars:
        axs[0].errorbar(dataset.times_rv[instrument] - 2458000, dataset.data_rv[instrument] - mu, \
                     yerr=dataset.errors_rv[instrument], fmt='o',
                     markeredgewidth=.75,
                     mec=colors[i], ecolor=colors[i], elinewidth=1.5, mfc='white', \
                     ms=3, label=aux.label(instrument), zorder=10)

        # Plot original errorbars + jitter (added in quadrature):
        axs[0].errorbar(dataset.times_rv[instrument] - 2458000, dataset.data_rv[instrument] - mu, \
                     yerr=np.sqrt(dataset.errors_rv[instrument] ** 2 + jitter ** 2), fmt='o', \
                     mec=colors[i], ecolor=colors[i], elinewidth=1.5, mfc='white',
                     ms=3, alpha=0.5, zorder=8)

        # plot residuals
        real_model = results.rv.evaluate(instrument, t=dataset.times_rv[instrument], all_samples=True)
        axs[1].errorbar(dataset.times_rv[instrument] - 2458000,
                        dataset.data_rv[instrument] - real_model,
                        yerr=dataset.errors_rv[instrument], fmt='o', \
                        markeredgewidth=.75,
                        mec=colors[i], ecolor=colors[i], elinewidth=1.5, mfc='white', \
                        ms=3, zorder=10)

        # and the error bars for jitter
        axs[1].errorbar(dataset.times_rv[instrument] - 2458000,
                        dataset.data_rv[instrument] - real_model,
                        yerr=np.sqrt(dataset.errors_rv[instrument] ** 2 + jitter ** 2), fmt='o', \
                        mec=colors[i], ecolor=colors[i], elinewidth=1.5, mfc='white',
                        ms=3, alpha=0.5, zorder=8)

    # Plot Keplerian model and CIs:
    axs[0].fill_between(model_times - 2458000, up68, low68,
                     color='cornflowerblue', alpha=0.5, zorder=5, label='model')
    axs[0].fill_between(model_times - 2458000, up95, low95,
                    color='cornflowerblue', alpha=0.3, zorder=6)
    axs[0].plot(model_times - 2458000, keplerian, color='black', zorder=7, lw=1)

   # # plt.title('Log-evidence: {:.2f} $\pm$ {:.2f}'.format(results.posteriors['lnZ'], \
   # #                                                                       results.posteriors['lnZerr']))
    axs[0].set_xlim([min_time - 2458000, max_time - 2458000])
    axs[0].set_ylabel('RV [m/s]')
    axs[1].axhline(0., ls='--', lw=2, color='gray')
    axs[1].set_xlabel('time [BJD - 2458000]')
    axs[1].set_ylabel('residuals [m/s]')
    axs[0].legend(loc='lower left', ncol=99, bbox_to_anchor=(0., 1.),
                  frameon=False, columnspacing=1.6)
    fig.align_ylabels()
    return fig, axs


def plot_phasedRV(results):
    """ plot phase-folded RV time series."""
    if isinstance(results, tuple):
        # sometimes, juliet.fit returns a tuple
        results = results[0]

    posteriors = results.posteriors
    # print(results.data.priors)
    # quit()
    dataset = results.data

    numbering_planets_rv = dataset.numbering_rv_planets
    num_planets = len(numbering_planets_rv)
    instruments_rv = dataset.inames_rv



    min_time, max_time = np.min([np.min(dataset.times_rv[k]) for k in instruments_rv]) - 4, \
                         np.max([np.max(dataset.times_rv[k]) for k in instruments_rv]) + 4
    model_rv_times = np.linspace(min_time, max_time, int((max_time - min_time) * 5))

    plots = {}
    for inst in instruments_rv:
        keplerian_model, kep_up68, kep_low68, components = results.rv.evaluate(inst,
                                                                               t=model_rv_times,
                                                                               return_err=True, alpha=0.68,
                                                                               return_components=True, )
        mu = np.median(posteriors['posterior_samples']['mu_{}'.format(inst)])
        keplerian_model -= mu
        kep_up68 -= mu
        kep_low68 -= mu


        for i_rv in numbering_planets_rv:
            # To plot the phased rv we need the median period and time-of-transit center:
            try:
                P = np.median(results.posteriors['posterior_samples']['P_p{}'.format(i_rv)])
            except KeyError:
                P = dataset.priors['P_p{}'.format(i_rv)]['hyperparameters']
            try:
                t0 = np.median(results.posteriors['posterior_samples']['t0_p{}'.format(i_rv)])
            except KeyError:
                t0 = dataset.priors['t0_p{}'.format(i_rv)]['hyperparameters']
            # Get phases:
            # Now plot the model for planet pl. First get phases of the model:
            phases_model = np.linspace(-0.5, 0.5, 1000)
            model_times = phases_model * P + t0

            try:
                model_rv, kep_up68, kep_low68, model_components = results.rv.evaluate(inst, t=model_times, \
                                                                                      return_components=True, \
                                                                                      return_err=True, alpha=0.68)
                _, kep_up95, kep_low95, _ = results.rv.evaluate(inst, t=model_times, \
                                                                return_components=True, \
                                                                return_err=True, alpha=0.95)
                _, kep_up99, kep_low99, _ = results.rv.evaluate(inst, t=model_times, \
                                                                return_components=True, \
                                                                return_err=True, alpha=0.99)
            except:
                model_rv, kep_up68, kep_low68, model_components = results.rv.evaluate(inst, t=model_times, \
                                                                                      return_components=True, \
                                                                                      GPregressors=model_times, \
                                                                                      return_err=True, alpha=0.68)
                _, kep_up95, kep_low95, _ = results.rv.evaluate(inst, t=model_times, \
                                                                return_components=True, \
                                                                GPregressors=model_times, \
                                                                return_err=True, alpha=0.95)
                _, kep_up99, kep_low99, _ = results.rv.evaluate(inst, t=model_times, \
                                                                return_components=True, \
                                                                GPregressors=model_times, \
                                                                return_err=True, alpha=0.99)

            fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2]})

            # Plot phased model:
            axs[0].plot(phases_model, model_components['p{}'.format(i_rv)], color='black', alpha=1, lw=1, zorder=3)
            axs[0].fill_between(phases_model, kep_up68 - model_components['mu'], kep_low68 - model_components['mu'], \
                             color='cornflowerblue', alpha=0.6, zorder=1, label='model')
            axs[0].fill_between(phases_model, kep_up95 - model_components['mu'], kep_low95 - model_components['mu'], \
                             color='cornflowerblue', alpha=0.3, zorder=1)
            # axs[0].fill_between(phases_model, kep_up99 - model_components['mu'], kep_low99 - model_components['mu'], \
            #                  color='cornflowerblue', alpha=0.4, zorder=1)

            # Plot the data
            for color, inst in zip(colors_rv, instruments_rv):
                phases_data = juliet.get_phases(dataset.times_rv[inst], P, t0)  # on the data time
                # Extract jitters:
                # Evaluate the median jitter for the instrument:
                try:
                    jitter = np.median(posteriors['posterior_samples']['sigma_w_' + inst])
                except:
                    jitter = 0.0
                mu = np.median(posteriors['posterior_samples']['mu_{}'.format(inst)])

                # Plot data with the full model *minus* planet n substracted, so we see the Keplerian of planet
                # pl imprinted on the data. For this, evaluate model in the data-times first:
                c_model, c_components = results.rv.evaluate(inst, t=dataset.times_rv[inst], \
                                                            all_samples=True, return_components=True, \
                                                            GPregressors=dataset.times_rv[inst])

                # Plot original data with original errorbars:
                axs[0].errorbar(phases_data, dataset.data_rv[inst] - (c_model - c_components['p{}'.format(i_rv)]),
                             yerr=dataset.errors_rv[inst], fmt='o',
                             markeredgewidth=.75,
                             mec=color, ecolor=color, elinewidth=1.5, mfc='white',
                             ms=3, label=aux.label(inst), zorder=10)

                # Plot original errorbars + jitter (added in quadrature):
                axs[0].errorbar(phases_data, dataset.data_rv[inst] - (c_model - c_components['p{}'.format(i_rv)]), \
                             yerr=np.sqrt(dataset.errors_rv[inst] ** 2 + jitter ** 2), fmt='o', \
                     mec=color, ecolor=color, elinewidth=1.5, mfc='white',
                     ms=3, alpha=0.5, zorder=8)

                # plot residuals
                axs[1].errorbar(phases_data, dataset.data_rv[inst] - c_model, \
                             yerr=dataset.errors_rv[inst], fmt='o', \
                        markeredgewidth=.75,
                        mec=color, ecolor=color, elinewidth=1.5, mfc='white', \
                        ms=3, zorder=10)

                # and the error bars for jitter
                axs[1].errorbar(phases_data, dataset.data_rv[inst] - c_model, \
                             yerr=np.sqrt(dataset.errors_rv[inst] ** 2 + jitter ** 2), fmt='o', \
                        mec=color, ecolor=color, elinewidth=1.5, mfc='white',
                        ms=3, alpha=0.5, zorder=8)

            # try:
            #     axs[0].set_title('P = {:.5f} t0 = {:.5f}'.format(P, t0))
            # except:
            #     axs[0].set_title('P = {:.5f} t0 = {:.5f}'.format(np.median(P), np.median(t0)))
            leg = axs[0].legend(loc='lower left', ncol=99, bbox_to_anchor=(0., 1.),
                                frameon=False, columnspacing=1.6)

            axs[1].axhline(y=0, ls='--', lw=1, color='k', alpha=0.5)
            # Define limits, labels:
            axs[0].set_ylabel('RV [m/s]')
            axs[1].set_ylabel('residuals [m/s]')
            axs[1].set_xlabel('orbital phase')
            axs[1].set_xlim([-0.5, 0.5])
            axs[1].set_xlim([-0.5, 0.5])
            # axs[0].yaxis.set_tick_params(labelsize=fontsize_rv_ticks)
            # axs[1].xaxis.set_tick_params(labelsize=fontsize_rv_ticks)
            # axs[1].yaxis.set_tick_params(labelsize=fontsize_rv_ticks)
            # axs[0].set_ylim([-20,20])
            axs[0].minorticks_on()
            axs[1].minorticks_on()
            axs[1].minorticks_on()
        plots[inst] = (fig, axs)
    return plots



def plot_Teq_theta(a, e, L, fig=None, ax=None, albedo=0., emissivity=1.,
                   beta=1., **kwargs):
    """plot equilibrium temperature as a function of true anomaly theta."""
    if ax is None:
        fig, ax = plt.subplots()#figsize=plotstyle.set_size())
    theta = np.linspace(0., 2*np.pi, 200)
    ax.plot(theta, aux.Teq(L, aux.r_of_theta(theta, a, e), albedo, emissivity,
                           beta), **kwargs)
    ax.set_xlabel('true anomaly [rad]')
    ax.set_ylabel('equilibrium temperature [K]')
    return fig, ax



def plot_periodograms(activityFile, plot_dir, results, saveFig=True):
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
    saveFig : boolean
        flag to save the figure as pdf

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

    try:
        transit_per = np.median(results.posteriors['posterior_samples']['P_p1'])
    except KeyError:
        transit_per = None

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
    annotOffsets = [-.10, 0, .10]
    axs[0].plot(rv_frequency, rv_power)
    for ind in range(len(rv_faps)):
        axs[0].axhline(rv_faps[ind], xmax=0.81,
            label = labels[ind], lw=1.5, linestyle = ltype[ind], c='black')
        axs[0].annotate(labels[ind], xy=[.815, rv_faps[ind]], va='center',
                        xytext=[.86, rv_faps[1] + annotOffsets[ind]], size=10,
                        xycoords=('axes fraction', 'data'), arrowprops=dict(arrowstyle="-"))
    if transit_per is not None:
        axs[0].axvline(1/transit_per,  lw=1.5, linestyle='dashed', color='C1')
        axs[0].annotate('P = {:.2f} d'.format(transit_per), [1/transit_per, 1.05], color='C1',
                    ha='center', xycoords=('data', 'axes fraction'))
    # axs[0].set_xscale('log')
    # axs[0].set_xlabel('Frequency [1/d]')
    axs[0].set_ylabel('power')
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
    if transit_per is not None:
        axs[1].axvline(1/transit_per, lw=1.5, linestyle='dashed', color='C1')
    # axs[1].set_xscale('log')
    # axs[1].set_xlabel('Period [d]')
    axs[1].set_ylabel('power')
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
    if transit_per is not None:
        axs[2].axvline(1/transit_per, lw=1.5, linestyle='dashed', color='C1')
    # axs[2].set_xscale('log')
    # axs[2].set_xlabel('Period [d]')
    axs[2].set_ylabel('power')
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
    if transit_per is not None:
        axs[3].axvline(1/transit_per, lw=1.5, linestyle='dashed', color='C1')
    # axs[3].set_xscale('log')
    # axs[3].set_xlabel('Period [d]')
    axs[3].set_ylabel('power')
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
    if transit_per is not None:
        axs[4].axvline(1/transit_per, lw=1.5, linestyle='dashed', color='C1')
    # axs[4].set_xscale('log')
    # axs[4].set_xlabel('Period [d]')
    axs[4].set_xlabel('Frequency [1/d]')
    axs[4].set_ylabel('power')
    axs[4].annotate(r'He I', xy=(0, 1.01), xytext=(.02, .84), size=15, bbox=bbox_props,
                               ha='left', va='center', xycoords='axes fraction', textcoords='axes fraction')

    # some eye candy
    [ax.set_xlim(left=0.005) for ax in axs]
    # [ax.set_ylim([0,.9]) for ax in axs]
    [ax.tick_params(direction='in', top=True, right=True) for ax in axs]
    # fig.subplots_adjust(hspace = .03, wspace=0.4)
    plt.show()
    if saveFig:
        fig.savefig(plot_dir + '/periodograms.pdf')

    # reset mpl settings
    font = {'size'   : 11}
    mpl.rc('font', **font)

    return fig, axs


def plot_RV_BS(activityFile, plot_dir, results):
    """
    Plot a scatter plot RV vs bisector span, color-coded by orbital phase.

    Parameters
    ----------
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
    ax : axis object
        contains axis object with the plot
    """

    if not 'P_p1' in results.posteriors['posterior_samples']:
        # for no-planet fits
        return None, None
    feros_dat = np.genfromtxt(activityFile, names=True)
    feros_dat = pd.DataFrame(feros_dat).replace(-999, np.nan)
    P = np.median(results.posteriors['posterior_samples']['P_p1'])
    t0 = np.median(results.posteriors['posterior_samples']['t0_p1'])
    feros_dat.loc[:, 'phase'] = juliet.utils.get_phases(feros_dat.BJD_OUT, P, t0)

    fig, ax = plt.subplots()  # figsize=plotstyle.set_size())
    ax.errorbar(feros_dat.RV, feros_dat.BS, xerr=feros_dat.RV_E, yerr=feros_dat.BS_E, c='k', fmt='o',
                lw=1.8, zorder=0, markeredgewidth=1.8, elinewidth=1.5)
    sc = ax.scatter(feros_dat.RV, feros_dat.BS, c=feros_dat.phase, s=100, zorder=10, cmap='twilight',
                    vmin=-.5, vmax=.5, edgecolor='white')
    cbar = fig.colorbar(sc, label='orbital phase')
    cbar.set_ticks(np.arange(-.5, .51, .25))
    cbar.set_ticklabels(['$-\pi$', '$-\pi/2$', '0', '$-\pi/2$', '$\pi$'])
    ax.set_xlabel('RV [km/s]')
    ax.set_ylabel('bisector span [km/s]')
    fig.savefig(plot_dir + '/RV-BS.pdf')
    return fig, ax


def plot_phasecurve(t, flux_planet, results, fig=None, ax=None, **kwargs):
    """ plot the phase curve."""

    # extract period, t_0 from posteriors, transform to phase space
    P = np.median(results.posteriors['posterior_samples']['P_p1'])
    t0 = np.median(results.posteriors['posterior_samples']['t0_p1'])
    t = juliet.utils.get_phases(t, P, t0)

    if ax is None:
        fig, ax = plt.subplots()
    # ax.plot(t - 2458000, flux_planet*1e6)
    ax.plot(t, flux_planet*1e6, **kwargs)
    # ax.set_xlabel("time [BJD - 2458000]")
    ax.set_xlabel("orbital phase")
    ax.set_ylabel("relative planetary flux [ppm]");
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    return fig, ax


def plot_periodEcc(TEPCat, P, e, P_err=None, e_err=None, M=None, color=None):
    """
    Plot a scatter plot period vs ecc, size-coded by planet mass.

    Parameters
    ----------
    TEPCat : pandas DataFrame
        TEPCat catalog of well-characterized planets
    color : string
        column name for color-coding
    ...

    Returns
    --------
    fig : matplotlib figure
        figure containing the plot
    ax : axis object
        contains axis object with the plot

    Notes
    -----
    column 'errup2' must be upper error of eccentricity.
    Jup, this is ugly.
    """
    c = TEPCat.copy()
    # exclude planets with upper limits on ecc and with negative ecc entries
    c = c[~((c.e == 0) & (c.errup2 > 0) | (c.e < 0))]
    c = c[c.M_b > 0]
    print('showing {} planets.'.format(len(c)))

    fig, ax = plt.subplots()  # figsize=set_size(width=513/2))

    # plot TEPCat catalog, optional color-code
    if color is not None:
        color = c[color]

    sc = ax.scatter(c.Period, c.e, s=2*c.M_b, c=color)

    # plot planet candidate
    try:
        #         ax.hlines(e, P_err[0], P_err[1])
        #         ax.vlines(P, e_err[0], e_err[1])
        ax.scatter(P, e, s=2*M)  # , label='TIC 237913194b')
        annotColor = 'black'
        _ = ax.annotate(r'TIC 237913194b', xy=(P, e), xytext=(P/2, 13/10*e), fontsize=8, va='center',
                        color=annotColor, ha='center', arrowprops=dict(arrowstyle='->',
                                                                       connectionstyle='arc3,rad=-0.2',
                                                                       color=annotColor))
    except:
        pass

    # make a legend with different marker sizes
    # Create some sizes and some labels.
    sizes = [10**i for i in range(-1, 3)]
    dummy = [-1 for i in range(len(sizes))]
    for i in range(len(sizes)):
        ax.scatter(dummy, dummy, s=2*sizes[i], label=sizes[i], c='C0')
    plt.legend(title='$\mathrm{M_p} \,\, [\mathrm{M_{Jup}}]$')

    if color is not None:
        cbar = fig.colorbar(sc, label='$T_\mathrm{eff}$')

    ax.set_xscale('log')
    ax.set_ylim(0, 1)
    #     ax.set_ylim(0,.85)
    ax.set_xlim(1, 100)
    ax.set_xlabel('period [d]')
    ax.set_ylabel('eccentricity')
    return fig, ax
