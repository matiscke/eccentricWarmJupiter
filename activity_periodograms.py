
import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from popsyntools import plotstyle

try:
    from astropy.timeseries import LombScargle
except ModuleNotFoundError:
    # the timeserie module moved at astropy 3.1 -> 3.2
    from astropy.stats import LombScargle
# plt.ion()

activityFile = 'data/TIC237913194_activity.dat'
plot_dir = 'plots/'
transit_per =  15.168914

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
    ax : matplotlib axis
        axis object with the plot
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
    axs[0].set_ylabel('power')
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
    axs[4].axvline(1/transit_per, lw=1.5, linestyle='dashed', color='C1')
    # axs[4].set_xscale('log')
    # axs[4].set_xlabel('Period [d]')
    axs[4].set_xlabel('Frequency [1/d]')
    axs[4].set_ylabel('power')
    axs[4].annotate(r'He I', xy=(0, 1.01), xytext=(.02, .84), size=15, bbox=bbox_props,
                               ha='left', va='center', xycoords='axes fraction', textcoords='axes fraction')

    # some eye candy
    [ax.set_xlim([0.005, 0.3]) for ax in axs]
    # [ax.set_ylim([0,.9]) for ax in axs]
    [ax.tick_params(direction='in', top=True, right=True) for ax in axs]
    # fig.subplots_adjust(hspace = .03, wspace=0.4)
    plt.show()
    fig.savefig(plot_dir + 'periodograms.pdf')

    return fig, ax


# import pickle
# out_folder = 'out/27_tess+chat+feros+GP'
# priors, params = get_priors(GP=True)
# times_lc, fluxes, fluxes_error, gp_times_lc = read_photometry(datafolder,
#                                                 plotPhot=False, outlierIndices=outlierIndices)
# times_rv, rvs, rvs_error = read_rv(datafolder)
#
# dataset = juliet.load(
#     priors=priors, t_lc=times_lc, y_lc=fluxes, yerr_lc=fluxes_error,
#     t_rv=times_rv, y_rv=rvs, yerr_rv=rvs_error,
#     GP_regressors_lc=gp_times_lc,
#     out_folder=out_folder, verbose=True)
# results = pickle.load(open(out_folder + '/results.pkl', 'rb'))
# plot_periodograms(activityFile, plot_dir, results)
# sys.exit(0)