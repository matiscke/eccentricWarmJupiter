"""
@author: Melissa Hobson
adapted by Martin Schlecker
"""

# =============================================================================
# Setup and imports
# =============================================================================
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


# ###### For nice plotting (Trifon)  ##############
# mpl.rcParams['axes.linewidth'] = 2.0 #set the value globally
# mpl.rcParams['xtick.major.pad']='8'
# mpl.rcParams['ytick.major.pad']='2'
#
# # set tick width
# mpl.rcParams['xtick.major.size'] = 8
# mpl.rcParams['xtick.major.width'] = 2
# mpl.rcParams['xtick.minor.size'] = 5
# mpl.rcParams['xtick.minor.width'] = 2
#
# mpl.rcParams['ytick.major.size'] = 8
# mpl.rcParams['ytick.major.width'] = 2
# mpl.rcParams['ytick.minor.size'] = 5
# mpl.rcParams['ytick.minor.width'] = 2
#
# mpl.rc('text',usetex=True)
font = {'size'   : 15}
mpl.rc('font', **font)
bbox_props = {'boxstyle':"round",
     'fc':"w",
     'edgecolor':"w",
     # 'ec':0.5,
     'alpha':0.9
}
#
# format_im = 'pdf'
# dpi = 300

###################3
results_dir = 'data/'
plot_dir = 'plots/'
transit_per =  15.168914

# =============================================================================
# Read data in
# =============================================================================

# FEROS data
feros_dat = np.genfromtxt(results_dir + 'TIC237913194_activity.dat',
              names=True)

# #HARPS data
# harps_dat = np.genfromtxt(results_dir+'harps/results_harps_all.txt',
#                           names=True, delimiter='\t')

# =============================================================================
# Periodograms - FEROS
# =============================================================================

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
rhk_feros = feros_dat['LOG_RHK']
rhk_e_feros = feros_dat['LOGRHK_E']

# create periodogram
rhk_ls = LombScargle(bjd_feros, rhk_feros, rhk_e_feros)
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
# Periodograms - HARPS
# =============================================================================

# #RV
# # variables
# bjd_harps = harps_dat['BJD_OUT']
# RV_harps = harps_dat['RV']
# RV_E_harps = harps_dat['RV_E']
# 
# # create periodogram
# rv_ls_harps = LombScargle(bjd_harps, RV_harps, RV_E_harps)
# rv_frequency_harps, rv_power_harps = rv_ls_harps.autopower(minimum_frequency=f_min,
#                                          maximum_frequency=f_max)
# 
# # Get FAP levels
# probabilities = [0.01, 0.005, 0.001]
# labels = ['FAP = 1%', 'FAP = 0.5%', 'FAP = 0.01%']
# ltype = ['solid', 'dashed', 'dotted']
# rv_faps_harps = rv_ls_harps.false_alarm_level(probabilities, method='bootstrap')
# 
# # H alpha
# # variables 
# ha_harps = harps_dat['HALPHA']
# ha_e_harps = harps_dat['HALPHA_E']
# 
# # create periodogram
# ha_ls_harps = LombScargle(bjd_harps, ha_harps, ha_e_harps)
# ha_frequency_harps, ha_power_harps = ha_ls_harps.autopower(minimum_frequency=f_min,
#                                          maximum_frequency=f_max)
# 
# # Get FAP levels
# ha_faps_harps = ha_ls_harps.false_alarm_level(probabilities, method='bootstrap')
# 
# # log Rhk
# # variables 
# rhk_harps = harps_dat['LOG_RHK']
# rhk_e_harps = harps_dat['LOGRHK_E']
# 
# # create periodogram
# rhk_ls_harps = LombScargle(bjd_harps, rhk_harps, rhk_e_harps)
# rhk_frequency_harps, rhk_power_harps = rhk_ls_harps.autopower(minimum_frequency=f_min,
#                                          maximum_frequency=f_max)
# 
# # Get FAP levels
# rhk_faps_harps = rhk_ls_harps.false_alarm_level(probabilities, method='bootstrap')
# 
# # Na II
# # variables 
# na_harps = harps_dat['NA_II']
# na_e_harps = harps_dat['NA_II_E']
# 
# # create periodogram
# na_ls_harps = LombScargle(bjd_harps, na_harps, na_e_harps)
# na_frequency_harps, na_power_harps = na_ls_harps.autopower(minimum_frequency=f_min,
#                                          maximum_frequency=f_max)
# 
# # Get FAP levels
# na_faps_harps = na_ls_harps.false_alarm_level(probabilities, method='bootstrap')
# 
# # He I
# # variables 
# he_harps = harps_dat['HE_I']
# he_e_harps = harps_dat['HE_I_E']
# 
# # create periodogram
# he_ls_harps = LombScargle(bjd_harps, he_harps, he_e_harps)
# he_frequency_harps, he_power_harps = he_ls_harps.autopower(minimum_frequency=f_min,
#                                          maximum_frequency=f_max)
# 
# # Get FAP levels
# he_faps_harps = he_ls_harps.false_alarm_level(probabilities, method='bootstrap')

# =============================================================================
# Plot the data
# =============================================================================

# plot everything together - FEROS
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
                    xycoords='axes fraction', arrowprops=dict(arrowstyle="-"))
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
# axs[4].suptitle('TOI-201 - FEROS results')






# some eye candy
[ax.set_xlim([0.005, 0.3]) for ax in axs]
# [ax.set_ylim([0,.9]) for ax in axs]
[ax.tick_params(direction='in', top=True, right=True) for ax in axs]
# fig.subplots_adjust(hspace = .03, wspace=0.4)
plt.show()
fig.savefig(plot_dir + 'periodograms.pdf')



# # plot everything together - HARPS
# plt.figure()
# # RV timeseries
# plt.subplot(2,3,1)
# plt.errorbar(bjd_harps, RV_harps, yerr=RV_E_harps, fmt='o')
# plt.set_xlabel('BJD')
# plt.set_ylabel('RV [km/s]')
# # RV periodogram
# plt.subplot(2,3,4)
# plt.plot(1/rv_frequency_harps, rv_power_harps)
# for ind in range(len(rv_faps_harps)):
# 	plt.axhline(rv_faps_harps[ind], np.min(1/rv_frequency_harps), np.max(1/rv_frequency_harps),
# 		label = labels[ind], lw=1.5, linestyle = ltype[ind], c='black')
# plt.vlines(1/transit_per, np.min(rv_power_harps), np.max(rv_power_harps), color='C1')
# # plt.set_xscale('log')
# plt.set_xlabel('Period [d]')
# plt.set_ylabel('Power')
# # Halpha timeseries
# plt.subplot(4, 3, 2)
# plt.errorbar(bjd_harps, ha_harps, yerr=ha_e_harps, fmt='o')
# plt.set_xlabel('BJD')
# plt.set_ylabel('H ALPHA')
# # Halpha periodogram
# plt.subplot(4, 3, 5)
# plt.plot(1/ha_frequency_harps, ha_power_harps)
# for ind in range(len(ha_faps_harps)):
# 	plt.axhline(ha_faps_harps[ind], np.min(1/ha_frequency_harps), np.max(1/ha_frequency_harps),
# 		label = labels[ind], lw=1.5, linestyle = ltype[ind], c='black')
# plt.vlines(1/transit_per, np.min(ha_power_harps), np.max(ha_power_harps), color='C1')
# # plt.set_xscale('log')
# plt.set_xlabel('Period [d]')
# plt.set_ylabel('Power')
# # log RHK timeseries
# plt.subplot(4, 3, 3)
# plt.errorbar(bjd_harps, rhk_harps, yerr=rhk_e_harps, fmt='o')
# plt.set_xlabel('BJD')
# plt.set_ylabel('LOG RHK')
# # log Rhk periodogram
# plt.subplot(4, 3, 6)
# plt.plot(1/rhk_frequency_harps, rhk_power_harps)
# for ind in range(len(rhk_faps_harps)):
# 	plt.axhline(rhk_faps_harps[ind], np.min(1/rhk_frequency_harps), np.max(1/rhk_frequency_harps),
# 		label = labels[ind], lw=1.5, linestyle = ltype[ind], c='black')
# plt.vlines(1/transit_per, np.min(rhk_power_harps), np.max(rhk_power_harps), color='C1')
# # plt.set_xscale('log')
# plt.set_xlabel('Period [d]')
# plt.set_ylabel('Power')
# # Na II timeseries
# plt.subplot(4, 3, 8)
# plt.errorbar(bjd_harps, na_harps, yerr=na_e_harps, fmt='o')
# plt.set_xlabel('BJD')
# plt.set_ylabel('NA II')
# # Na II periodogram
# plt.subplot(4, 3, 11)
# plt.plot(1/na_frequency_harps, na_power_harps)
# for ind in range(len(na_faps_harps)):
# 	plt.axhline(na_faps_harps[ind], np.min(1/na_frequency_harps), np.max(1/na_frequency_harps),
# 		label = labels[ind], lw=1.5, linestyle = ltype[ind], c='black')
# plt.vlines(1/transit_per, np.min(na_power), np.max(na_power), color='C1')
# # plt.set_xscale('log')
# plt.set_xlabel('Period [d]')
# plt.set_ylabel('Power')
# # HeI timeseries
# plt.subplot(4, 3, 9)
# plt.errorbar(bjd_harps, he_harps, yerr=he_e_harps, fmt='o')
# plt.set_xlabel('BJD')
# plt.set_ylabel('HE I')
# # HeI periodogram
# plt.subplot(4, 3, 12)
# plt.plot(1/he_frequency_harps, he_power_harps)
# for ind in range(len(he_faps_harps)):
# 	plt.axhline(he_faps_harps[ind], np.min(1/he_frequency_harps), np.max(1/he_frequency_harps),
# 		label = labels[ind], lw=1.5, linestyle = ltype[ind], c='black')
# plt.vlines(1/transit_per, np.min(he_power_harps), np.max(he_power_harps), color='C1')
# # plt.set_xscale('log')
# plt.set_xlabel('Period [d]')
# plt.set_ylabel('Power')
# plt.suptitle('TOI-201 - HARPS results')
# 
# # =============================================================================
# # Periodograms - Joint data - absolute
# # =============================================================================
# 
# #RV
# # variables
# bjd_joint = np.hstack((bjd_feros, bjd_harps))
# bjd_sort = np.argsort(bjd_joint)
# bjd_joint = bjd_joint[bjd_sort]
# RV_joint = np.hstack((RV_feros, RV_harps))
# RV_joint = RV_joint[bjd_sort]
# RV_E_joint = np.hstack((RV_E_feros, RV_E_harps))
# RV_E_joint = RV_E_joint[bjd_sort]
# 
# # create periodogram
# rv_ls_joint = LombScargle(bjd_joint, RV_joint, RV_E_joint)
# rv_frequency_joint, rv_power_joint = rv_ls_joint.autopower(minimum_frequency=f_min,
#                                          maximum_frequency=f_max)
# 
# # Get FAP levels
# probabilities = [0.01, 0.005, 0.001]
# labels = ['FAP = 1%', 'FAP = 0.5%', 'FAP = 0.01%']
# ltype = ['solid', 'dashed', 'dotted']
# rv_faps_joint = rv_ls_joint.false_alarm_level(probabilities, method='bootstrap')
# 
# # H alpha
# # variables 
# ha_joint = np.hstack((ha_feros, ha_harps))
# ha_joint = ha_joint[bjd_sort]
# ha_e_joint = np.hstack((ha_e_feros, ha_e_harps))
# ha_e_joint = ha_e_joint[bjd_sort]
# 
# # create periodogram
# ha_ls_joint = LombScargle(bjd_joint, ha_joint, ha_e_joint)
# ha_frequency_joint, ha_power_joint = ha_ls_joint.autopower(minimum_frequency=f_min,
#                                          maximum_frequency=f_max)
# 
# # Get FAP levels
# ha_faps_joint = ha_ls_joint.false_alarm_level(probabilities, method='bootstrap')
# 
# # log Rhk
# # variables 
# rhk_joint = np.hstack((rhk_feros, rhk_harps))
# rhk_joint = rhk_joint[bjd_sort]
# rhk_e_joint = np.hstack((rhk_e_feros, rhk_e_harps))
# rhk_e_joint = rhk_e_joint[bjd_sort]
# 
# # create periodogram
# rhk_ls_joint = LombScargle(bjd_joint, rhk_joint, rhk_e_joint)
# rhk_frequency_joint, rhk_power_joint = rhk_ls_joint.autopower(minimum_frequency=f_min,
#                                          maximum_frequency=f_max)
# 
# # Get FAP levels
# rhk_faps_joint = rhk_ls_joint.false_alarm_level(probabilities, method='bootstrap')
# 
# # Na II
# # variables 
# na_joint = np.hstack((na_feros, na_harps))
# na_joint = na_joint[bjd_sort]
# na_e_joint = np.hstack((na_e_feros, na_e_harps))
# na_e_joint = na_e_joint[bjd_sort]
# 
# # create periodogram
# na_ls_joint = LombScargle(bjd_joint, na_joint, na_e_joint)
# na_frequency_joint, na_power_joint = na_ls_joint.autopower(minimum_frequency=f_min,
#                                          maximum_frequency=f_max)
# 
# # Get FAP levels
# na_faps_joint = na_ls_joint.false_alarm_level(probabilities, method='bootstrap')
# 
# # He I
# # variables 
# he_joint = np.hstack((he_feros, he_harps))
# he_joint = he_joint[bjd_sort]
# he_e_joint = np.hstack((he_e_feros, he_e_harps))
# he_e_joint = he_e_joint[bjd_sort]
# 
# # create periodogram
# he_ls_joint = LombScargle(bjd_joint, he_joint, he_e_joint)
# he_frequency_joint, he_power_joint = he_ls_joint.autopower(minimum_frequency=f_min,
#                                          maximum_frequency=f_max)
# 
# # Get FAP levels
# he_faps_joint = he_ls_joint.false_alarm_level(probabilities, method='bootstrap')
# 
# 
# # plot everything together - joint
# plt.figure()
# # RV timeseries
# plt.subplot(2,3,1)
# plt.errorbar(bjd_feros, RV_feros, yerr=RV_E_feros, fmt='o', label = 'FEROS')
# plt.errorbar(bjd_harps, RV_harps, yerr=RV_E_harps, fmt='o', label = 'HARPS')
# plt.set_xlabel('BJD')
# plt.set_ylabel('RV [km/s]')
# plt.legend()
# # RV periodogram
# plt.subplot(2,3,4)
# plt.plot(1/rv_frequency_joint, rv_power_joint)
# for ind in range(len(rv_faps_joint)):
# 	plt.axhline(rv_faps_joint[ind], np.min(1/rv_frequency_joint), np.max(1/rv_frequency_joint),
# 		label = labels[ind], lw=1.5, linestyle = ltype[ind], c='black')
# plt.vlines(1/transit_per, np.min(rv_power_joint), np.max(rv_power_joint), color='C1')
# # plt.set_xscale('log')
# plt.set_xlabel('Period [d]')
# plt.set_ylabel('Power')
# # Halpha timeseries
# plt.subplot(4, 3, 2)
# plt.errorbar(bjd_feros, ha_feros, yerr=ha_e_feros, fmt='o', label = 'FEROS')
# plt.errorbar(bjd_harps, ha_harps, yerr=ha_e_harps, fmt='o', label = 'HARPS')
# plt.set_xlabel('BJD')
# plt.set_ylabel('H ALPHA')
# plt.legend()
# # Halpha periodogram
# plt.subplot(4, 3, 5)
# plt.plot(1/ha_frequency_joint, ha_power_joint)
# for ind in range(len(ha_faps_joint)):
# 	plt.axhline(ha_faps_joint[ind], np.min(1/ha_frequency_joint), np.max(1/ha_frequency_joint),
# 		label = labels[ind], lw=1.5, linestyle = ltype[ind], c='black')
# plt.vlines(1/transit_per, np.min(ha_power_joint), np.max(ha_power_joint), color='C1')
# # plt.set_xscale('log')
# plt.set_xlabel('Period [d]')
# plt.set_ylabel('Power')
# # log RHK timeseries
# plt.subplot(4, 3, 3)
# plt.errorbar(bjd_feros, rhk_feros, yerr=rhk_e_feros, fmt='o', label = 'FEROS')
# plt.errorbar(bjd_harps, rhk_harps, yerr=rhk_e_harps, fmt='o', label = 'HARPS')
# plt.set_xlabel('BJD')
# plt.set_ylabel('LOG RHK')
# plt.legend()
# # log Rhk periodogram
# plt.subplot(4, 3, 6)
# plt.plot(1/rhk_frequency_joint, rhk_power_joint)
# for ind in range(len(rhk_faps_joint)):
# 	plt.axhline(rhk_faps_joint[ind], np.min(1/rhk_frequency_joint), np.max(1/rhk_frequency_joint),
# 		label = labels[ind], lw=1.5, linestyle = ltype[ind], c='black')
# plt.vlines(1/transit_per, np.min(rhk_power_joint), np.max(rhk_power_joint), color='C1')
# # plt.set_xscale('log')
# plt.set_xlabel('Period [d]')
# plt.set_ylabel('Power')
# # Na II timeseries
# plt.subplot(4, 3, 8)
# plt.errorbar(bjd_feros, na_feros, yerr=na_e_feros, fmt='o', label = 'FEROS')
# plt.errorbar(bjd_harps, na_harps, yerr=na_e_harps, fmt='o', label = 'HARPS')
# plt.set_xlabel('BJD')
# plt.set_ylabel('NA II')
# plt.legend()
# # Na II periodogram
# plt.subplot(4, 3, 11)
# plt.plot(1/na_frequency_joint, na_power_joint)
# for ind in range(len(na_faps_joint)):
# 	plt.axhline(na_faps_joint[ind], np.min(1/na_frequency_joint), np.max(1/na_frequency_joint),
# 		label = labels[ind], lw=1.5, linestyle = ltype[ind], c='black')
# plt.vlines(1/transit_per, np.min(na_power), np.max(na_power), color='C1')
# # plt.set_xscale('log')
# plt.set_xlabel('Period [d]')
# plt.set_ylabel('Power')
# # HeI timeseries
# plt.subplot(4, 3, 9)
# plt.errorbar(bjd_feros, he_feros, yerr=he_e_feros, fmt='o', label = 'FEROS')
# plt.errorbar(bjd_harps, he_harps, yerr=he_e_harps, fmt='o', label = 'HARPS')
# plt.set_xlabel('BJD')
# plt.set_ylabel('HE I')
# plt.legend()
# # HeI periodogram
# plt.subplot(4, 3, 12)
# plt.plot(1/he_frequency_joint, he_power_joint)
# for ind in range(len(he_faps_joint)):
# 	plt.axhline(he_faps_joint[ind], np.min(1/he_frequency_joint), np.max(1/he_frequency_joint),
# 		label = labels[ind], lw=1.5, linestyle = ltype[ind], c='black')
# plt.vlines(1/transit_per, np.min(he_power_joint), np.max(he_power_joint), color='C1')
# # plt.set_xscale('log')
# plt.set_xlabel('Period [d]')
# plt.set_ylabel('Power')
# plt.suptitle('TOI-201 - joint results')
# 
# # =============================================================================
# # Periodograms - Joint data - median corrected
# # =============================================================================
# 
# # Mask  HARPS point xx
# mask = np.ones(len(bjd_harps), dtype=bool)                                                                                                                                           
# # mask[xx] = False
# 
# #RV
# # time variable and sorted mask
# bjd_joint_m = np.hstack((bjd_feros, bjd_harps[mask]))
# bjd_sort_m = np.argsort(bjd_joint_m)
# bjd_joint_m = bjd_joint_m[bjd_sort_m]
# # remove RV median
# RV_feros_m = RV_feros - np.median(RV_feros)
# RV_harps_m = RV_harps[mask] - np.median(RV_harps[mask])
# RV_joint_m = np.hstack((RV_feros_m, RV_harps_m))
# RV_joint_m = RV_joint_m[bjd_sort_m]
# RV_E_joint_m = np.hstack((RV_E_feros, RV_E_harps[mask]))
# RV_E_joint_m = RV_E_joint_m[bjd_sort_m]
# 
# # create periodogram
# rv_ls_joint_m = LombScargle(bjd_joint_m, RV_joint_m, RV_E_joint_m)
# rv_frequency_joint_m, rv_power_joint_m = rv_ls_joint_m.autopower(minimum_frequency=f_min,
#                                          maximum_frequency=f_max)
# 
# # Get FAP levels
# probabilities = [0.01, 0.005, 0.001]
# labels = ['FAP = 1%', 'FAP = 0.5%', 'FAP = 0.01%']
# ltype = ['solid', 'dashed', 'dotted']
# rv_faps_joint_m = rv_ls_joint_m.false_alarm_level(probabilities, method='bootstrap')
# 
# # H alpha - not median correcting because seem to be at same level
# # variables 
# ha_joint_m = np.hstack((ha_feros, ha_harps[mask]))
# ha_joint_m = ha_joint_m[bjd_sort_m]
# ha_e_joint_m = np.hstack((ha_e_feros, ha_e_harps[mask]))
# ha_e_joint_m = ha_e_joint_m[bjd_sort_m]
# 
# # create periodogram
# ha_ls_joint_m = LombScargle(bjd_joint_m, ha_joint_m, ha_e_joint_m)
# ha_frequency_joint_m, ha_power_joint_m = ha_ls_joint_m.autopower(minimum_frequency=f_min,
#                                          maximum_frequency=f_max)
# 
# # Get FAP levels
# ha_faps_joint_m = ha_ls_joint_m.false_alarm_level(probabilities, method='bootstrap')
# 
# # log Rhk - not median correcting because seem to be at same level
# # variables 
# rhk_joint_m = np.hstack((rhk_feros, rhk_harps[mask]))
# rhk_joint_m = rhk_joint_m[bjd_sort_m]
# rhk_e_joint_m = np.hstack((rhk_e_feros, rhk_e_harps[mask]))
# rhk_e_joint_m = rhk_e_joint_m[bjd_sort_m]
# 
# # create periodogram
# rhk_ls_joint_m = LombScargle(bjd_joint_m, rhk_joint_m, rhk_e_joint_m)
# rhk_frequency_joint_m, rhk_power_joint_m = rhk_ls_joint_m.autopower(minimum_frequency=f_min,
#                                          maximum_frequency=f_max)
# 
# # Get FAP levels
# rhk_faps_joint_m = rhk_ls_joint_m.false_alarm_level(probabilities, method='bootstrap')
# 
# # Na II - median correcting as seems to be offset!
# # variables 
# na_feros_m = na_feros - np.median(na_feros)
# na_harps_m = na_harps[mask] - np.median(na_harps[mask])
# na_joint_m = np.hstack((na_feros_m, na_harps_m))
# na_joint_m = na_joint_m[bjd_sort_m]
# na_e_joint_m = np.hstack((na_e_feros, na_e_harps[mask]))
# na_e_joint_m = na_e_joint_m[bjd_sort_m]
# 
# # create periodogram
# na_ls_joint_m = LombScargle(bjd_joint_m, na_joint_m, na_e_joint_m)
# na_frequency_joint_m, na_power_joint_m = na_ls_joint_m.autopower(minimum_frequency=f_min,
#                                          maximum_frequency=f_max)
# 
# # Get FAP levels
# na_faps_joint_m = na_ls_joint_m.false_alarm_level(probabilities, method='bootstrap')
# 
# # He I - not median correcting because seem to be the same level
# # variables 
# he_joint_m = np.hstack((he_feros, he_harps[mask]))
# he_joint_m = he_joint_m[bjd_sort_m]
# he_e_joint_m = np.hstack((he_e_feros, he_e_harps[mask]))
# he_e_joint_m = he_e_joint_m[bjd_sort_m]
# 
# # create periodogram
# he_ls_joint_m = LombScargle(bjd_joint_m, he_joint_m, he_e_joint_m)
# he_frequency_joint_m, he_power_joint_m = he_ls_joint_m.autopower(minimum_frequency=f_min,
#                                          maximum_frequency=f_max)
# 
# # Get FAP levels
# he_faps_joint_m = he_ls_joint_m.false_alarm_level(probabilities, method='bootstrap')
# 
# 
# # plot everything together - joint
# plt.figure()
# # RV timeseries
# plt.subplot(2,3,1)
# plt.errorbar(bjd_feros, RV_feros_m*1000, yerr=RV_E_feros*1000, fmt='o', label = 'FEROS')
# plt.errorbar(bjd_harps[mask], RV_harps_m*1000, yerr=RV_E_harps[mask]*1000, fmt='o', label = 'HARPS')
# plt.set_xlabel('BJD')
# plt.set_ylabel('RV -median (RV)[m/s]')
# plt.legend()
# # RV periodogram
# plt.subplot(2,3,4)
# plt.plot(1/rv_frequency_joint_m, rv_power_joint_m)
# for ind in range(len(rv_faps_joint_m)):
# 	plt.axhline(rv_faps_joint_m[ind], np.min(1/rv_frequency_joint_m), np.max(1/rv_frequency_joint_m),
# 		label = labels[ind], lw=1.5, linestyle = ltype[ind], c='black')
# plt.vlines(1/transit_per, np.min(rv_power_joint_m), np.max(rv_power_joint_m), color='C1')
# # plt.set_xscale('log')
# plt.set_xlabel('Period [d]')
# plt.set_ylabel('Power')
# # Halpha timeseries
# plt.subplot(4, 3, 2)
# plt.errorbar(bjd_feros, ha_feros, yerr=ha_e_feros, fmt='o', label = 'FEROS')
# plt.errorbar(bjd_harps[mask], ha_harps[mask], yerr=ha_e_harps[mask], fmt='o', label = 'HARPS')
# plt.set_xlabel('BJD')
# plt.set_ylabel('H ALPHA')
# plt.legend()
# # Halpha periodogram
# plt.subplot(4, 3, 5)
# plt.plot(1/ha_frequency_joint_m, ha_power_joint_m)
# for ind in range(len(ha_faps_joint)):
# 	plt.axhline(ha_faps_joint[ind], np.min(1/ha_frequency_joint_m), np.max(1/ha_frequency_joint_m),
# 		label = labels[ind], lw=1.5, linestyle = ltype[ind], c='black')
# plt.vlines(1/transit_per, np.min(ha_power_joint_m), np.max(ha_power_joint_m), color='C1')
# # plt.set_xscale('log')
# plt.set_xlabel('Period [d]')
# plt.set_ylabel('Power')
# # log RHK timeseries
# plt.subplot(4, 3, 3)
# plt.errorbar(bjd_feros, rhk_feros, yerr=rhk_e_feros, fmt='o', label = 'FEROS')
# plt.errorbar(bjd_harps[mask], rhk_harps[mask], yerr=rhk_e_harps[mask], fmt='o', label = 'HARPS')
# plt.set_xlabel('BJD')
# plt.set_ylabel('LOG RHK')
# plt.legend()
# # log Rhk periodogram
# plt.subplot(4, 3, 6)
# plt.plot(1/rhk_frequency_joint_m, rhk_power_joint_m)
# for ind in range(len(rhk_faps_joint_m)):
# 	plt.axhline(rhk_faps_joint_m[ind], np.min(1/rhk_frequency_joint_m), np.max(1/rhk_frequency_joint_m),
# 		label = labels[ind], lw=1.5, linestyle = ltype[ind], c='black')
# plt.vlines(1/transit_per, np.min(rhk_power_joint_m), np.max(rhk_power_joint_m), color='C1')
# # plt.set_xscale('log')
# plt.set_xlabel('Period [d]')
# plt.set_ylabel('Power')
# # Na II timeseries
# plt.subplot(4, 3, 8)
# plt.errorbar(bjd_feros, na_feros_m, yerr=na_e_feros, fmt='o', label = 'FEROS')
# plt.errorbar(bjd_harps[mask], na_harps_m, yerr=na_e_harps[mask], fmt='o', label = 'HARPS')
# plt.set_xlabel('BJD')
# plt.set_ylabel('Na II - median(Na II)')
# plt.legend()
# # Na II periodogram
# plt.subplot(4, 3, 11)
# plt.plot(1/na_frequency_joint_m, na_power_joint_m)
# for ind in range(len(na_faps_joint_m)):
# 	plt.axhline(na_faps_joint_m[ind], np.min(1/na_frequency_joint_m), np.max(1/na_frequency_joint_m),
# 		label = labels[ind], lw=1.5, linestyle = ltype[ind], c='black')
# plt.vlines(1/transit_per, np.min(na_power_joint_m), np.max(na_power_joint_m), color='C1')
# # plt.set_xscale('log')
# plt.set_xlabel('Period [d]')
# plt.set_ylabel('Power')
# # HeI timeseries
# plt.subplot(4, 3, 9)
# plt.errorbar(bjd_feros, he_feros, yerr=he_e_feros, fmt='o', label = 'FEROS')
# plt.errorbar(bjd_harps[mask], he_harps[mask], yerr=he_e_harps[mask], fmt='o', label = 'HARPS')
# plt.set_xlabel('BJD')
# plt.set_ylabel('HE I')
# plt.legend()
# # HeI periodogram
# plt.subplot(4, 3, 12)
# plt.plot(1/he_frequency_joint_m, he_power_joint_m)
# for ind in range(len(he_faps_joint_m)):
# 	plt.axhline(he_faps_joint_m[ind], np.min(1/he_frequency_joint_m), np.max(1/he_frequency_joint_m),
# 		label = labels[ind], lw=1.5, linestyle = ltype[ind], c='black')
# plt.vlines(1/transit_per, np.min(he_power_joint_m), np.max(he_power_joint_m), color='C1')
# # plt.set_xscale('log')
# plt.set_xlabel('Period [d]')
# plt.set_ylabel('Power')
# plt.suptitle('TOI-201 - joint results - median-subtracted')
#


sys.exit(0)