import numpy as np 
from astropy.io import fits 
import matplotlib.pyplot as plt
import os
import juliet 
import pickle 
# import utils_planets
import sys
# sys.path.append("/Users/kossakowski/anaconda3/lib/python3.6/site-packages/juliet/") # this is to import utils_planets, which could live there
import utils_planets
# import latextable

m_s = 1.03 # solar mass
m_s_e = .06
r_s = 1.088 # solar raddi
r_s_e = .012

m_s_dist = np.random.normal(loc=m_s, scale=m_s_e,size=10000) # solar mass
r_s_dist = np.random.normal(loc=r_s, scale=r_s_e,size=10000) # solar radii

solarrad2m = 6.957e8 
m_s_dist *= 1.98847e30 # kg
r_s_dist *= solarrad2m # m

rho_s = m_s_dist / (4/3. * np.pi * r_s_dist**3) # solar density kgm-3
# plt.hist(rho_s, bins=100)
# plt.show()
val,valup,valdown = juliet.utils.get_quantiles(rho_s)
print('Stellar density: ', val,valup-val,val-valdown)

def plot_posteriors(posteriors, out_folder):
	num_samps = len(posteriors['posterior_samples']['unnamed'])

	if not os.path.exists(out_folder+'/posteriorplots/'):


		os.mkdir(out_folder+'/posteriorplots/')
	for k in posteriors['posterior_samples'].keys():
	    if k != 'unnamed':
	        val,valup,valdown = juliet.utils.get_quantiles(posteriors['posterior_samples'][k])
	        print(k,':',val,' + ',valup-val,' - ',val-valdown)
	        fig = plt.figure(figsize=(10,7))

	        plt.hist(posteriors['posterior_samples'][k],bins=int(len(posteriors['posterior_samples'][k])/50),color='k',histtype='step',lw=2)
	        plt.axvline(x=val,color='cornflowerblue',lw=1.5,ls='--',label='{} = {:.5}'.format(k, val))
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

input_folder = ''
out_folder = 'test08_DsSkript' ### CHANGE THIS
instruments_lc = ['TESSERACT+TESS', 'CHAT+i']
instruments_rv = ['FEROS'] ### CHANGE THIS
colors_rv = ['orangered','cornflowerblue', 'purple', 'forestgreen'] ### CHANGE THIS
datafolder = '/Users/schlecker/WINE/TIC237913194/data/'
times = {}
# times['TESSERACT+TESS'] = np.loadtxt(datafolder+'photometry.tess.txt', unpack=True, usecols=(0))

# cat TESS.lc.dat ELSAUCE_Rc.dat > phot.dat

# datafolder = '/data/beegfs/astro-storage/groups/henning/kossakowski/toi732/data/'

times_lc, fluxes, fluxes_error = {},{},{}
for inst in instruments_lc:
    times_lc[inst], fluxes[inst], fluxes_error[inst] = np.loadtxt(datafolder+inst.lower()+'.lc.dat', unpack=True, usecols=(0,1,2))
#     plt.scatter(times_lc[inst]-2450000, fluxes[inst],s=1)
#     plt.title(inst)
#     plt.show()
# quit()
times_rv, rvs, rvs_error = {},{},{}

# gp_times = {}
for inst in instruments_rv:
    times_rv[inst], rvs[inst], rvs_error[inst] = np.loadtxt(datafolder+inst.lower()+'.rv.dat', unpack=True, usecols=(0,1,2))
    # gp_times[inst] = times[inst]

# cat_str = 'cat'
# for inst in instruments_lc:
#     cat_str += ' {}.lc.dat'.format(datafolder+inst.lower())
#
# cat_str += ' > {}phot.dat'.format(datafolder)
# print(cat_str)
# os.system(cat_str)

# quit()

# Name of the parameters to be fit:
params = [\
        # planet 1
        'P_p1','t0_p1',\
        'r1_p1','r2_p1',\
        'sesinomega_p1','secosomega_p1',\

        # # planet 2
        # 'P_p2','t0_p2',\
        # 'r1_p2','r2_p2','sesinomega_p2','secosomega_p2',\

        'rho', \

        # TESS
		'q1_TESSERACT+TESS','q2_TESSERACT+TESS',\
        'sigma_w_TESSERACT+TESS', 'mflux_TESSERACT+TESS', 'mdilution_TESSERACT+TESS',\
        'GP_sigma_TESSERACT+TESS', 'GP_timescale_TESSERACT+TESS',\

        # # CHAT+i
        'q1_CHAT+i', \
        'sigma_w_CHAT+i', 'mflux_CHAT+i', 'mdilution_CHAT+i',\

        # # RV planetary
        'K_p1',\
        ]

# Distributions:
dists = [\
        # planet 1
        'normal', 'normal',\
        # 'fixed','fixed',\
        'uniform','uniform',
        'uniform','uniform',\

        # # planet 2
        # 'normal', 'normal',\
        # # 'fixed','fixed',\
        # 'uniform','uniform','fixed','fixed',\

        'normal', \

        # TESS
        'uniform','uniform',\
        'loguniform', 'normal', 'fixed', \
        'loguniform','loguniform',\

        # # CHAT+i
        'uniform',\
        'loguniform', 'normal', 'fixed', \

        # # RV planetary
        'uniform',\
        ]

# Hyperparameters
hyperps = [\
        # planet 1
        [15.16, 0.2], [2458319.17, 0.2],\
        # 15.1688886522,2458319.1496740049,\
        # 4.05176, 2451446.489,\
        [0.,1], [0.,1.],
        [-1, 1], [-1, 1],\


        # # planet 2
        # [12.254218,0.005], [2458546.848389,0.01],\
        # # 4.05176, 2451446.489,\
        # [0.,1], [0.,1.], 0.0, 0.0,\

        [1120,110], \
        # [100., 10000.], \

        #
        # TESS
        [0., 1.], [0., 1.],\
        [1e-5,1e5], [0.0,0.1], 1.0, \
        [1e-7, 1e-2], [1e-3, 3],\

        # # CHAT+i
        [0., 1.], \
        [1e-5,1e5], [0.0,0.1], 1.0, \

        # # RV planetary
        [0.,30.],\
        ]

params_gp = [\
            'GP_sigma_CARMENES-VIS_ESPRESSO_HIRES', \
            # 'GP_Gamma_CARMENES-VIS_ESPRESSO_HIRES', \
            # 'GP_alpha_CARMENES-VIS_ESPRESSO_HIRES', \
            # 'GP_Prot_CARMENES-VIS_ESPRESSO_HIRES',\
            'GP_alpha0_CARMENES-VIS_ESPRESSO_HIRES',\
            ]

dists_gp = [\
            'loguniform', \
            # 'loguniform', \
            # 'loguniform', \
            # 'loguniform',\
            'loguniform',\
            ]
hyperps_gp = [\
              [1e-2,30],\
              # [1e-5,1e2],\
              # [1e-5,1e1],\
              # [1,100],\
              [1e-5,10.0],\
              ]

params_feros = ['mu_FEROS', 'sigma_w_FEROS']
dists_feros = ['uniform', 'loguniform']
hyperps_feros = [[-10,40],[1e-5,1.]]

params+= params_feros
dists+=dists_feros
hyperps+=hyperps_feros

# params_lt = ['rv_intercept', 'rv_slope']
# dists_lt = ['uniform', 'uniform']
# hyperps_lt = [[-100,100],[-100,100.]]
# params+= params_lt
# dists+=dists_lt
# hyperps+=hyperps_lt

# params += params_gp
# dists += dists_gp
# hyperps += hyperps_gp

priors = {}
for param, dist, hyperp in zip(params, dists, hyperps):
    priors[param] = {}
    priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp
print(priors)
# dataset = juliet.load(priors=priors, t_lc = times, y_lc = fluxes, \
#                       yerr_lc = fluxes_error, \
#                       # GP_regressors_lc=times, \
#                       # GPlceparamfile = datafolder+'TESS.eparams.dat',\
#                       # linear_regressors_lc = linear_regressors_lc, \
#                       ld_laws = 'ELSAUCE-linear',\
#                       # ld_laws = 'TESS-quadratic, ELSAUCE-linear, LCOGTZ-linear, MONET-linear, TRAPPIST-linear, TUG-linear, ULMT-linear',\
#                       out_folder = out_folder, \
#                       verbose=True)

if input_folder != '':
    dataset = juliet.load(input_folder=input_folder, GPlceparamfile = input_folder+'/lc_eparams.dat')
else:
    dataset = juliet.load(priors=priors, \
                      t_lc = times_lc, y_lc = fluxes, yerr_lc = fluxes_error, \
                      # lcfilename='{}photometry.tess.txt'.format(datafolder),\
                      # rvfilename='{}FEROS.rv.dat'.format(datafolder),\
                      # GP_regressors_lc=times, \
                      # GPlceparamfile = datafolder+'tess.eparams.dat',\
                      t_rv = times_rv, y_rv = rvs, yerr_rv = rvs_error,\
                      # GP_regressors_rv=times_rv, \
                      # ld_laws = 'TESSERACT+TESS-quadratic,CHAT+i-linear',\
                      out_folder = out_folder, \
                      verbose=True)


# results = dataset.fit(use_dynesty = True, dynamic = True, dynesty_bound = 'multi', dynesty_sample='rwalk', \
#             n_live_points = 1000)
results = dataset.fit(use_dynesty = True)
utils_planets.get_planetaryparams(dataset, results, mass_star=[m_s,m_s_e],
                                  radius_star=[r_s,r_s_e], teff_star=[5788,80], vsini_star=2.18)
posteriors = results.posteriors

plot_posteriors(posteriors, dataset.out_folder)
#quit()

### Printing out the LATEX tables (IN PROGRESS FORGIVE ME IF NOT SO GOOD YET)
# latextable.print_prior_table(dataset)
# latextable.print_posterior_table(dataset, posteriors['posterior_samples'], precision=2)
# quit()
###

### If you want to select certain samples
# idx_samples = np.where(results.posteriors['posterior_samples']['r1_p1']<0.4)[0]
# new_posteriors = {}
# new_posteriors['posterior_samples'] = {}
# for k in results.posteriors['posterior_samples'].keys():
#     # We copy all the keys but the "unnamed" one --- we don't need that one.
#     new_posteriors['posterior_samples'][k] = results.posteriors['posterior_samples'][k][idx_samples]
# posteriors = new_posteriors
###

#################
#################
#################


#################
#################
#################
# print(dataset.data_lc)
# print(dataset.errors_lc)
# print(dataset.times_lc)
# print(dataset.n_transiting_planets)
# print(dataset.numbering_transiting_planets)
# quit()

numbering_planets_transit = dataset.numbering_transiting_planets
instruments_lc = dataset.inames_lc
if instruments_lc is not None:
    for inst in instruments_lc:
        # Plot:
        fig,(ax1,ax2) = plt.subplots(2,1,sharex=True, gridspec_kw = {'height_ratios':[5,2]}, facecolor='w',figsize=(10,4))

        # gs = gridspec.GridSpec(2, 1, height_ratios=[2,1])
        nsamples = 1000

        fil = dataset.out_folder+'/lc_model_{}.pkl'.format(inst)
        if os.path.isfile(fil):
            model = pickle.load(open(fil,'rb'))
            lc_model,lc_up68,lc_low68,components = model['lc_model'], model['lc_up68'], model['lc_low68'], model['components']
        else:
            model = {}

            model['lc_model'], model['lc_up68'], model['lc_low68'], model['components'] = lc_model,lc_up68,lc_low68,components = results.lc.evaluate(inst, \
                parameter_values=posteriors['posterior_samples'], \
                t=dataset.times_lc[inst],\
                nsamples=nsamples,\
                return_err=True,\
                return_components=True,\
                GPregressors=dataset.times_lc[inst])
            model['nsamples']=nsamples

            pickle.dump(model,open(fil,'wb'))
        # transit_model = results.lc.model[inst]['deterministic']
        # gp_model = results.lc.model[inst]['GP']
        # print(components)
        # First the data and the model on top:
        # ax1 = plt.subplot(gs[0])
        ax1.errorbar(dataset.times_lc[inst]-2458000, dataset.data_lc[inst], dataset.errors_lc[inst],fmt='.',alpha=0.8, color='cornflowerblue')
        ax1.plot(dataset.times_lc[inst]-2458000, lc_model, color='black', zorder=100)
        ax1.set_ylabel('Relative flux')
        ax1.set_title('{0} | Log-evidence: {1:.3f} $\pm$ {2:.3f}'.format(inst, results.posteriors['lnZ'],\
             results.posteriors['lnZerr']))
        ax1.set_xlim(np.min(dataset.times_lc[inst]-2458000),np.max(dataset.times_lc[inst]-2458000))
        ax1.minorticks_on()
        # ax1.xaxis.set_major_formatter(plt.NullFormatter())

        # Now the residuals:
        # ax2 = plt.subplot(gs[1])
        ax2.errorbar(dataset.times_lc[inst]-2458000, (dataset.data_lc[inst]-lc_model)*1e6, \
                   dataset.errors_lc[inst]*1e6,fmt='.',alpha=0.8,color='cornflowerblue')
        ax2.set_ylabel('Residuals (ppm)')
        ax2.set_xlabel('Time (BJD - 2458000)')
        ax2.set_xlim(np.min(dataset.times_lc[inst]-2458000),np.max(dataset.times_lc[inst]-2458000))
        plt.tight_layout()
        fig.subplots_adjust(hspace=0) # to make the space between rows smaller
        ax2.minorticks_on()
        plt.savefig(dataset.out_folder+'/phot_vs_time_{}.pdf'.format(inst), dpi=700)
        # plt.show()
        plt.close(fig)

        for i_transit in numbering_planets_transit:
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

            fig,(ax2,ax3) = plt.subplots(2,1,sharex=True, gridspec_kw = {'height_ratios':[5,2]}, facecolor='w',figsize=(8,6))
            phases_lc = juliet.utils.get_phases(dataset.times_lc[inst], P, t0)
            idx = np.argsort(phases_lc)

            c_model, c_components = results.lc.evaluate(inst, t = dataset.times_lc[inst], \
                                                        # all_samples=True, \
                                                        return_components = True,\
                                                        GPregressors=dataset.times_lc[inst])

            # print(gp_model)
            # print(c_model)
            # c_model = linear + transit = linear + p1 + p2 + ... + pn
            # we want: data - linear - p2 - ... - pn = data - linear - (transit - p1) = data - (c_model - p1)
            ax2.errorbar(phases_lc, dataset.data_lc[inst] - (c_model - c_components['p{}'.format(i_transit)]) - gp_model, \
                yerr = dataset.errors_lc[inst], fmt = '.',\
                alpha = 0.1)

            # Plot transit-only (divided by mflux) model:
            ax2.plot(phases_lc[idx],c_components['p{}'.format(i_transit)][idx] - c_components['lm'][idx], color='black',zorder=10)
            ax3.axhline(y=0, ls='--', color='k', alpha=0.5)
            # ax2.yaxis.set_major_formatter(plt.NullFormatter())
            try:
                ax2.set_title('P = {:.5f} t0 = {:.5f}'.format(P, t0))
            except:
                ax2.set_title('P = {:.5f} t0 = {:.5f}'.format(np.median(P), np.median(t0)))

            ax2.set_ylabel('Relative flux')
            ax2.set_xlim([-0.1,0.1]) ### CHANGE THIS
            # ax2.set_ylim([0.9985,1.0015]) ### CHANGE THIS
            ax2.minorticks_on()
            ax3.errorbar(phases_lc, (dataset.data_lc[inst]-c_model -gp_model)*1e6, \
                         yerr = dataset.errors_lc[inst], fmt = '.', alpha = 0.1)
            ax3.set_ylabel('Residuals (ppm)')
            ax3.set_xlabel('Phases')
            plt.tight_layout()
            fig.subplots_adjust(hspace=0) # to make the space between rows smaller
            ax3.minorticks_on()
            plt.savefig(dataset.out_folder+'/phased_lc_{}_pl{}.pdf'.format(inst,i_transit), dpi=700)
            # plt.show()
            plt.close(fig)

#quit()





















############
############
############
## Plotting RV
############
############
############
# Plot CARMENES data with model. 

numbering_planets_rv = dataset.numbering_rv_planets
num_planets = len(numbering_planets_rv)
instruments_rv = dataset.inames_rv
instruments = instruments_rv
print('RV instruments: ', instruments_rv)

nsamples = 1000

fil = dataset.out_folder+'/rv_model.pkl'
if os.path.isfile(fil):
    model = pickle.load(open(fil,'rb'))
    model_rv_times = model['model_rv_times']
    keplerian_model, kep_up68, kep_low68, components = model['keplerian_model'], model['kep_up68'], model['kep_low68'], model['components']
else:

    # Plot data with the full model *minus* planet n substracted, so we see the Keplerian of planet
    # pl imprinted on the data. For this, evaluate model in the data-times first:
    model = {}
    min_time, max_time = np.min([np.min(dataset.times_rv[k]) for k in instruments]) - 1,\
                      np.max([np.max(dataset.times_rv[k]) for k in instruments]) + 1

    model_rv_times = np.linspace(min_time,max_time,5000)
    model['model_rv_times'] = model_rv_times
    # Now evaluate the model in those times, and substract the systemic-velocity to
    # get the Keplerian signal:
    model['keplerian_model'], model['kep_up68'], model['kep_low68'], model['components'] = keplerian_model, kep_up68, kep_low68, components = results.rv.evaluate(instruments[0], \
                    t = model_rv_times, \
                    return_err=True, \
                    nsamples=nsamples,\
                    return_components = True, \
                    GPregressors=model_rv_times)
    model['nsamples']=nsamples

    pickle.dump(model,open(fil,'wb'))

keplerian_model -= np.median(posteriors['posterior_samples']['mu_{}'.format(instruments[0])])

# Now plot the (systematic-velocity corrected) RVs:
fig,(ax,ax2) = plt.subplots(2,1,sharex=True, gridspec_kw = {'height_ratios':[5,2]}, facecolor='w',figsize=(12,5))
for color,inst in zip(colors_rv,instruments):
    # Evaluate the median jitter for the instrument:
    try:
        jitter = np.median(posteriors['posterior_samples']['sigma_w_'+inst])
    except:
        jitter = 0.0
    # Evaluate the median systemic-velocity:
    mu = np.median(posteriors['posterior_samples']['mu_'+inst])
    keplerian_rv = results.rv.evaluate(inst, t = dataset.times_rv[inst], GPregressors=dataset.times_rv[inst]) - mu
    ax.fill_between(model_rv_times-2458000,kep_up68-mu, kep_low68-mu,\
                 color='whitesmoke',alpha=1,zorder=1)

    # Plot original data with original errorbars:
    ax.errorbar(dataset.times_rv[inst]-2458000, dataset.data_rv[inst]-mu,\
                yerr = dataset.errors_rv[inst],\
                fmt='o',\
                mec=color, ecolor=color, elinewidth=2, mfc = 'white', \
                ms = 7, alpha=0.5, label=inst, zorder=10)

    # Plot original errorbars + jitter (added in quadrature):
    ax.errorbar(dataset.times_rv[inst]-2458000, dataset.data_rv[inst]-mu,\
                yerr = np.sqrt(dataset.errors_rv[inst]**2+jitter**2), \
                fmt='o',\
                mec=color, ecolor=color, mfc = 'white', \
                alpha = 0.5, zorder=5)

    ax2.errorbar(dataset.times_rv[inst]-2458000, dataset.data_rv[inst]-keplerian_rv-mu,\
                yerr = dataset.errors_rv[inst],\
                fmt='o',\
                mec=color, ecolor=color, elinewidth=2, mfc = 'white', \
                ms = 7, alpha=0.5, zorder=10)
    ax2.errorbar(dataset.times_rv[inst]-2458000,dataset.data_rv[inst]-keplerian_rv-mu,\
                yerr = np.sqrt(dataset.errors_rv[inst]**2+jitter**2),\
                fmt='o',\
                mec=color, ecolor=color, mfc = 'white', \
                alpha=0.5, zorder=5)

    rv_residuals_fil = fil = dataset.out_folder+'/rv_residuals_{}.dat'.format(inst)
    fout = open(rv_residuals_fil,'w')
    fout.write('# time RVresid RVe inst\n')
    for t,residrv,rve in zip(dataset.times_rv[inst], dataset.data_rv[inst]-keplerian_rv-mu, dataset.errors_rv[inst]):
        fout.write('{0:} {1:} {2:} {3:}\n'.format(t,residrv,rve,inst))
    fout.close()    

# Plot Keplerian model:
ax.plot(model_rv_times-2458000, keplerian_model, color='black',zorder=3)
ax2.axhline(y=0, ls='--', color='k', alpha=0.5)


ax.minorticks_on()
ax.legend(loc='upper right',fontsize=12)
ax.set_ylabel('RV (m/s)')
plot_title = '{} planet ('.format(num_planets)
for i_rv in numbering_planets_rv:
    try:
        per = np.median(results.posteriors['posterior_samples']['P_p{}'.format(i_rv)])
    except:
        per = dataset.priors['P_p{}'.format(i_rv)]['hyperparameters']
    plot_title += '{:.3g}d, '.format(per)
plot_title = plot_title[:-2]
plot_title += ') Model'

ax.set_title('{0} | Log-evidence: {1:.3f} $\pm$ {2:.3f}'.format(plot_title, results.posteriors['lnZ'],\
       results.posteriors['lnZerr']))
# plt.ylim([-20,20])
ax.set_xlim([np.min(model_rv_times)-2458000, np.max(model_rv_times)-2458000])
ax2.set_xlim([np.min(model_rv_times)-2458000, np.max(model_rv_times)-2458000])
ax2.set_xlabel('Time (BJD - '+str(2458000)+')')
ax2.set_ylabel('Residual RV (m/s)')
ax2.minorticks_on()
plt.tight_layout()
fig.subplots_adjust(hspace=0) # to make the space between rows smaller

plt.savefig(dataset.out_folder+'/rv_vs_time.pdf', dpi=700)
# plt.show()
plt.close(fig)
# quit()
### PHASED



# Evaluate RV model --- use all the posterior samples, also extract model components:
# rv_model, kep_up68, kep_low68, components = results.rv.evaluate(instruments[0], t = model_rv_times, \
#                                             # all_samples = True, \
#                                             return_err = True,\
#                                             return_components = True, GPregressors=model_rv_times)

# # Substract CARMENES systemic RV from rv_model:
# rv_model -= components['mu']
rv_model = keplerian_model
for i_rv in numbering_planets_rv:
    # To plot the phased rv we need the median period and time-of-transit center:
    try:
        P = results.posteriors['posterior_samples']['P_p{}'.format(i_rv)]
    except KeyError:
        P = dataset.priors['P_p{}'.format(i_rv)]['hyperparameters']
    try:
        t0 = results.posteriors['posterior_samples']['t0_p{}'.format(i_rv)]
    except KeyError:
        t0 = dataset.priors['t0_p{}'.format(i_rv)]['hyperparameters']
    # Get phases:
    # Now plot the model for planet pl. First get phases of the model:
    phases_model = juliet.get_phases(model_rv_times, P, t0) # on the model time

    fig,(ax,ax2) = plt.subplots(2,1,sharex=True, gridspec_kw = {'height_ratios':[5,2]}, facecolor='w',figsize=(8,6))

    # Plot phased model:
    idx = np.argsort(phases_model)
    ax.plot(phases_model[idx], components['p{}'.format(i_rv)][idx], color='black', alpha=0.8, lw = 2, zorder=3)      

    # Plot the data
    for color,inst in zip(colors_rv,instruments):
        phases_data = juliet.get_phases(dataset.times_rv[inst], P, t0) # on the data time
        # Extract jitters:
        jitter = np.median(posteriors['posterior_samples']['sigma_w_'+inst])
        mu = np.median(posteriors['posterior_samples']['mu_{}'.format(inst)])
            
        fil = dataset.out_folder+'/rv_phased_{}.pkl'.format(inst) 

        if os.path.isfile(fil):
            c = pickle.load(open(fil,'rb'))
            c_model, c_components = c['model'], c['components']
        else:

            # Plot data with the full model *minus* planet n substracted, so we see the Keplerian of planet
            # pl imprinted on the data. For this, evaluate model in the data-times first:
            c = {}
            c['model'], c['components'] = c_model, c_components = results.rv.evaluate(inst, t = dataset.times_rv[inst], \
                                                        all_samples=True, return_components = True,\
                                                        GPregressors=dataset.times_rv[inst])
            print(c_components.keys())

            pickle.dump(c,open(fil,'wb'))
        ax.errorbar(phases_data, dataset.data_rv[inst] - (c_model - c_components['p{}'.format(i_rv)]), \
            yerr = dataset.errors_rv[inst], fmt = 'o',\
            mec=color, ecolor=color, elinewidth=2, mfc = 'white', \
            ms = 7, alpha = 0.5, zorder=10, label=inst)
        ax.errorbar(phases_data, dataset.data_rv[inst] - (c_model - c_components['p{}'.format(i_rv)]), \
            yerr = np.sqrt(dataset.errors_rv[inst]**2+jitter**2), fmt = 'o',\
            mec=color, ecolor=color, mfc = 'white', \
            alpha = 0.3, zorder=5)
        # - (rv_model[idx] - components['p{}'.format(i_rv)][idx])
        ax.fill_between(phases_model[idx], kep_up68[idx] - (rv_model[idx] - components['p{}'.format(i_rv)][idx]) - mu, \
                kep_low68[idx] - (rv_model[idx] - components['p{}'.format(i_rv)][idx]) -  mu,\
                color='whitesmoke',alpha=1,zorder=1)
        # Plot original data with original errorbars:
        ax2.errorbar(phases_data, dataset.data_rv[inst] - c_model, \
            yerr = dataset.errors_rv[inst], fmt = 'o',\
            mec=color, ecolor=color, elinewidth=2, mfc = 'white', \
            ms = 7, alpha = 0.5, zorder=10)
        # Plot original errorbars + jitter (added in quadrature):
        ax2.errorbar(phases_data, dataset.data_rv[inst] - c_model, \
            yerr = np.sqrt(dataset.errors_rv[inst]**2+jitter**2), fmt = 'o',\
            mec=color, ecolor=color, mfc = 'white', \
            alpha = 0.3, zorder=5)

    try:
        ax.set_title('P = {:.5f} t0 = {:.5f}'.format(P, t0))
    except:
        ax.set_title('P = {:.5f} t0 = {:.5f}'.format(np.median(P), np.median(t0)))
    ax.legend(loc='upper right',fontsize=12)

    ax2.axhline(y=0, ls='--', color='k', alpha=0.5)
    # Define limits, labels:
    ax.set_ylabel('RV (m/s)')
    ax2.set_ylabel('RV residuals (m/s)')
    ax2.set_xlabel('Phases')
    ax.set_xlim([-0.5,0.5])
    ax2.set_xlim([-0.5,0.5])
    # ax.set_ylim([-20,20])
    ax.minorticks_on()
    ax2.minorticks_on()
    plt.tight_layout()
    fig.subplots_adjust(hspace=0) # to make the space between rows smaller


    plt.savefig(dataset.out_folder+'/phased_rv_pl{}.pdf'.format(i_rv), dpi=700)
    plt.show()
    plt.close(fig)


