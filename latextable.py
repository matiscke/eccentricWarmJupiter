""" modified from Diana Kossakowski's script.
"""

import numpy as np
import juliet
from math import log10, floor
import decimal

# https://exoplanetarchive.ipac.caltech.edu/docs/poet_calculations.html

G = 6.67408e-11 # m3 kg-1 s-2
solarrad2m = 6.957e8 # solar radii in meters
solarmass2kg = 1.9891e30 # solar mass in kg
earthrad2m = 6.371e6 # earth radii in meters
jupiterrad2m = 69.911e6 # jupiter radii in meters
earthmass2kg = 5.972e24 # earth mass in kg
jupitermass2kg = 1.898e27 # jupiet mass in kg
ergs2solarlum = 3.839e33 # erg s-1 = 1L solar
AU2m = 149600000000 # 1 AU = 149600000000m

#ms$^{-1}$
priors_dict = {\
'P': {'units': 'd', 'description': 'Period'},\
't0': {'units': 'd', 'description': 'Time of transit center'},\
'a': {'units': '??', 'description': '??'},\
'r1': {'units': '---', 'description': 'Parametrization for p and b'},\
'r2': {'units': '---', 'description': 'Parametrization for p and b'},\
'b': {'units': '---', 'description': 'Impact factor'},\
'p': {'units': '---', 'description': 'Planet-to-star ratio'},\
'K': {'units': 'm/s', 'description': 'Radial velocity semi-amplitude'},\
'ecc': {'units': '---', 'description': 'Orbital eccentricity'},\
'sesinomega': {'units': '---', 'description': 'Parametrization for $e$ and $\\omega$'},\
'secosomega': {'units': '---', 'description': 'Parametrization for $e$ and $\\omega$'},\
'esinomega': {'units': '---', 'description': 'Parametrization for $e$ and $\\omega$'},\
'ecosomega': {'units': '---', 'description': 'Parametrization for $e$ and $\\omega$'},\

# RV instrumental
'mu': {'units': 'm/s', 'description': 'Systemic velocity for '},\
'sigma_w': {'units': 'm/s', 'description': 'Extra jitter term for '},\

'rv_quad': {'units': 'm/s/d$^2$', 'description': 'Quadratic term for the RVs'},\
'rv_slope': {'units': 'm/s/d', 'description': 'Linear term for the RVs'},\
'rv_intercept': {'units': 'm/s', 'description': 'Intercept term for the Rvs'},\

# Photometry instrumental
'mdilution': {'units': '---', 'description': 'Dilution factor for '},\
'mflux': {'units': 'ppm', 'description': 'Relative flux offset for '},\
# 'sigma_w': {'units': 'ppm', 'description': 'Extra jitter term for '},\
'q1': {'units': '---', 'description': 'Linear limb-darkening parametrization'},\
'q2': {'units': '---', 'description': 'Quadratic limb-darkening parametrization'},\

# Theta terms
'theta0': {'units': '', 'description': 'Offset value applied to \\textcolor{red}{add}'},\
'theta1': {'units': '', 'description': 'Offset value applied to \\textcolor{red}{add}'},\
'theta2': {'units': '', 'description': 'Offset value applied to \\textcolor{red}{add}'},\
'theta3': {'units': '', 'description': 'Offset value applied to \\textcolor{red}{add}'},\
'theta4': {'units': '', 'description': 'Offset value applied to \\textcolor{red}{add}'},\
'theta5': {'units': '', 'description': 'Offset value applied to \\textcolor{red}{add}'},\
'theta6': {'units': '', 'description': 'Offset value applied to \\textcolor{red}{add}'},\

# Other
'rho': {'units': 'kg/m$^3$', 'description': 'Stellar density'},\

# GP
'GP_Prot': {'units': '\\textcolor{red}{add}', 'description': 'Rotational period component for the GP'},\
'GP_Gamma': {'units': '\\textcolor{red}{add}', 'description': 'Amplitude of periodic component for the GP'},\
'GP_sigma': {'units': 'm/s', 'description': 'Amplitude of the GP component'},\
'GP_alpha': {'units': '\\textcolor{red}{add}', 'description': 'Parametrization of the lengthscale of the GP'},\


}

dists_types = {'normal': '\\mathcal{N}', \
                'uniform': '\\mathcal{U}', \
                'loguniform': '\\mathcal{J}', \
                'jeffreys': '\\mathcal{J}', \
                'beta': '\\mathcal{B}', \
                'exponential': '\\Gamma', \
                'truncatednormal': '\\mathcal{N_T}'}#, 'fixed']

order_planetary = ['P', 't0', 'a', 'r1', 'r2', 'b', 'p', 'K', 'ecc', 'omega', 'sesinomega', 'secosomega', 'esinomega', 'ecosomega']
planet_names = ['', 'b', 'c', 'd', 'e', 'f'] # the 0th index is empty just to keep indexing normal later on
orderrv_lt = ['rv_quad', 'rv_slope', 'rv_intercept']
thetaterms = ['theta0', 'theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6']
orderinst_lc = np.append(['mdilution','mflux', 'sigma_w', 'q1', 'q2'], thetaterms)
orderinst_rv = np.append(['mu','sigma_w'], thetaterms)

linend = '\\\\[0.1 cm]' # end of a line

def print_data_table(dataset, type='rv'):
    out_folder = dataset.out_folder
    # if latex_fil = out_folder+'/data_table.tex'
    fout = open(latex_fil, 'w')

def produce_this(s_param, dataset, param, key, params_priors, inst):
    if dataset.priors[key]['distribution'] == 'fixed':
        s_param += '{} (fixed)'.format(dataset.priors[key]['hyperparameters'])
    elif dataset.priors[key]['distribution'] == 'exponential':
        s_param += '$'+dists_types[dataset.priors[key]['distribution']]+'('+str(dataset.priors[key]['hyperparameters'][0])+')'
        # s_param += ')$ & '

    elif dataset.priors[key]['distribution'] == 'truncatednormal':
        s_param += '$'+dists_types[dataset.priors[key]['distribution']]+'('+\
                    str(dataset.priors[key]['hyperparameters'][0])+','+\
                    str(dataset.priors[key]['hyperparameters'][1])+','+\
                    str(dataset.priors[key]['hyperparameters'][2])+','+\
                    str(dataset.priors[key]['hyperparameters'][3])+')'
        # s_param += ')$ & '
    else:
        if dataset.priors[key]['distribution'] not in dists_types:
            print('BUG_PRODUCE_THIS_FUNCTION')
            print(key)
            print(dataset.priors[key])
            print()
            quit()
        s_param += '$'+dists_types[dataset.priors[key]['distribution']]+'('+str(dataset.priors[key]['hyperparameters'][0])+','+\
                    str(dataset.priors[key]['hyperparameters'][1])
        if dataset.priors[key]['distribution'] in ['normal', 'loguniform', 'jeffreys']:
            s_param += '^2'
        s_param+=')'
    s_param += ' $ & '

    if param == 'sigma_w': s_param += 'ppm & ' # there already is a sigma_w for RV with ms-1
    else: s_param += priors_dict[param]['units']+' & '

    if param == 'q1' and 'q2' in params_priors: s_param += priors_dict['q2']['description'] # change from linear to quad
    else: s_param += priors_dict[param]['description']

    if priors_dict[param]['description'][-4:] == 'for ': s_param += inst 

    s_param += linend

    return s_param

def print_prior_table(dataset):

    out_folder = dataset.out_folder
    latex_fil = out_folder+'/prior_table.tex'
    fout = open(latex_fil, 'w')

    params_priors = np.array([i for i in dataset.priors.keys()]) # transform it to array in order to manipulate it within the function

    ## Beginning of the table
    tab_start = ['\\begin{table*}', '\\centering', '\\caption{Prior parameters}', '\\label{tab:priors}', '\\begin{tabular}{lccl}',\
                '\\hline', '\\hline', '\\noalign{\\smallskip}', 'Parameter name & Prior & Units & Description \\\\',\
                '\\noalign{\\smallskip}', '\\hline', '\\hline']

    for elem in tab_start:
        fout.write(elem+'\n')

    ## Stellar parameters first
    if 'rho' in params_priors:
        fout.write('\\noalign{\\smallskip}\n')
        fout.write('Stellar Parameters '+linend+' \n')
        fout.write('\\noalign{\\smallskip}\n')
        s_param = '~~~'
        s_param += '$\\rho_{*}$'
        s_param += '  & '
        s_param = produce_this(s_param=s_param, dataset=dataset, param='rho', key='rho', params_priors=params_priors, inst='')
        fout.write(s_param+'\n')
        params_priors = np.delete(params_priors, np.where(params_priors == 'rho')[0])
    ## Planet part first

    # Save information stored in the prior: the dictionary, number of transiting planets, 
    # number of RV planets, numbering of transiting and rv planets (e.g., if p1 and p3 transit 
    # and all of them are RV planets, numbering_transit = [1,3] and numbering_rv = [1,2,3]). 
    # Save also number of *free* parameters (FIXED don't count here).
    
    rv_planets = dataset.numbering_rv_planets
    transiting_planets = dataset.numbering_transiting_planets

    for pl in np.unique(np.append(rv_planets, transiting_planets)):
        fout.write('\\noalign{\\smallskip}\n')
        fout.write('Parameters for planet {} '.format(planet_names[pl])+linend+' \n')
        fout.write('\\noalign{\\smallskip}\n')

        for param in order_planetary:
            key = '{}_p{}'.format(param, pl)
            if key not in params_priors:
                continue
            s_param = '~~~'
            if param == 't0': s_param += '$t_{0,' + planet_names[pl] + '}$' 
            elif param == 'a': s_param += '$a_{' + planet_names[pl] + '}/R_*$' 
            elif param == 'r1': s_param += '$r_{1,' + planet_names[pl] + '}$'
            elif param == 'r2': s_param += '$r_{2,' + planet_names[pl] + '}$'
            elif param == 'p': s_param += '$R_{' + planet_names[pl] + '}/R_*$'
            elif param == 'b': s_param += '$b = (a_{' + planet_names[pl] + '}/R_*) \\cos (i_{'+ planet_names[pl] +'}) $'
            elif param == 'ecc': s_param += '$e_{' + planet_names[pl] + '}$'
            elif param == 'omega': s_param += '$\\omega_{' + planet_names[pl] + '}$'
            elif param == 'sesinomega': 
                s_param += '$S_{1,' + planet_names[pl] + '} = \\sqrt{e_'+planet_names[pl]+'}\\sin \\omega_'+planet_names[pl]+'$'
            elif param == 'secosomega':
                s_param += '$S_{2,' + planet_names[pl] + '} = \\sqrt{e_'+planet_names[pl]+'}\\cos \\omega_'+planet_names[pl]+'$'
            elif param == 'esinomega':
                s_param += '$S_{1,' + planet_names[pl] + '} = e_'+planet_names[pl]+'\\sin \\omega_'+planet_names[pl]+'$'
            elif param == 'ecosomega':
                s_param += '$S_{1,' + planet_names[pl] + '} = e_'+planet_names[pl]+'\\cos \\omega_'+planet_names[pl]+'$'
            else: s_param += '$' + param + '_{' + planet_names[pl] + '}$'
            s_param += '  & '

            s_param = produce_this(s_param=s_param, dataset=dataset, param=param, key=key, params_priors=params_priors, inst='')
            fout.write(s_param+'\n')
            params_priors = np.delete(params_priors, np.where(params_priors == key)[0])
    
    # Instruments
    instruments_rv = dataset.inames_rv
    instruments_lc = dataset.inames_lc

    if instruments_rv is not None and len(instruments_rv) > 0:
        fout.write('\\noalign{\\smallskip}\n')
        fout.write('RV instrumental parameters'+linend+'\n')
        fout.write('\\noalign{\\smallskip}\n')
        for inst in instruments_rv:
            for param in orderinst_rv:
                key = '{}_{}'.format(param, inst)
                if key not in params_priors: continue

                s_param = '~~~'

                if param == 'mu': s_param += '$\\mu'
                elif param == 'sigma_w': s_param += '$\\sigma'
                elif param[:5] == 'theta': s_param += '$\\theta_{'+str(param[-1])+','

                s_param += '_{\\textnormal{' + inst + '}}$  & '
                s_param = produce_this(s_param=s_param, dataset=dataset, param=param, key=key, params_priors=params_priors, inst=inst)
                fout.write(s_param+'\n')
                params_priors = np.delete(params_priors, np.where(params_priors == key)[0])

    # if there were linear/quadratic trends for the rv
    if 'rv_slope' in params_priors:
        for param in orderrv_lt:
            key = param
            if key not in params_priors: continue
            s_param = '~~~'

            if param == 'rv_quad':
                s_param += '$\\textnormal{RV}_{\\textnormal{quadratic}}$'
            elif param == 'rv_slope':
                s_param += '$\\textnormal{RV}_{\\textnormal{linear}}$'
            else:
                s_param += '$\\textnormal{RV}_{\\textnormal{intercept}}$'

            s_param += '  & '

            s_param = produce_this(s_param=s_param, dataset=dataset, param=param, key=key, params_priors=params_priors, inst='')

            fout.write(s_param+'\n')
            params_priors = np.delete(params_priors, np.where(params_priors == key)[0])

    # instruments_lc = ['TESSERACT+TESS']

    if instruments_lc is not None and len(instruments_lc) > 0:
        fout.write('\\noalign{\\smallskip}\n')
        fout.write('Photometry instrumental parameters'+linend+'\n')
        fout.write('\\noalign{\\smallskip}\n')
        for inst in instruments_lc:
            for param in orderinst_lc:
                key = '{}_{}'.format(param, inst)
                if key not in params_priors: continue
                s_param = '~~~'

                print('lc param "{}"'.format(param))
                if param == 'mdilution': s_param += '$D_{'
                elif param == 'mflux': s_param += '$M_{'
                elif param == 'sigma_w': s_param += '$\\sigma_{'
                elif param == 'q1': s_param += '$q_{1,'
                elif param == 'q2' and key in params_priors: s_param += '$q_{2,'
                elif param[:5] == 'theta' and key in params_priors: s_param += '$\\theta_{'+str(param[-1])+','


                s_param += '\\textnormal{' + inst + '}}$  & '
                s_param = produce_this(s_param=s_param, dataset=dataset, param=param, key=key, params_priors=params_priors, inst=inst)
                fout.write(s_param+'\n')
                params_priors = np.delete(params_priors, np.where(params_priors == key)[0])

            fout.write('\\noalign{\\medskip}\n')

    # print('prior leftover params', params_priors)

    gp_names = ['sigma', 'alpha', 'Gamma', 'Prot', 'B', 'L', 'C', 'timescale', 'rho', 'S0', 'Q', 'omega0']
    gp_names_latex = ['\\sigma', '\\alpha', '\\Gamma', 'P_{rot}', \
                    '\\textnormal{B}', '\\textnormal{L}', '\\textnormal{C}', '\\tau', \
                    '\\rho', 'S_0', '\\textnormal{Q}', '\\omega 0']

    if len(params_priors) > 1:
        fout.write('\\noalign{\\smallskip}\n')
        fout.write('Additional parameters{}\n'.format(linend))
        fout.write('\\noalign{\\smallskip}\n')

    for post in params_priors:
        s_param = '~~~'
        if post == 'unnamed': continue
        elif post[:2] == 'GP':
            pvector = post.split('_')
            gp_param = pvector[1]
            insts = pvector[2:]

            for i,elem in enumerate(gp_names):
                if elem == gp_param: 
                    gp_param_latex = gp_names_latex[i]
                    continue

            if gp_param not in gp_names: # most likely alpha0, alpha1
                gp_param_latex = '\\alpha_{}'.format(gp_param[-1])

            s_param += gp_param_latex + '$^{GP}_{\\textnormal{'
            for inst in insts:
                s_param += inst+','
            s_param = s_param[:-1] # to get rid of the last comma
            s_param += '}}$'
        else:
            s_param += '$' + post + '$'
        s_param += '  & '

        if dataset.priors[post]['distribution'] == 'fixed':
            s_param += '{}'.format(dataset.priors[post]['hyperparameters']) 
            s_param += ' (fixed)  & --- & \\textcolor{red}{add}'
        elif dataset.priors[post]['distribution'] == 'exponential':
            s_param += '$'+dists_types[dataset.priors[post]['distribution']]+'('+str(dataset.priors[post]['hyperparameters'])
            s_param += ')$ & --- & \\textcolor{red}{add}'

        elif dataset.priors[post]['distribution'] == 'truncatednormal':
            s_param += '$'+dists_types[dataset.priors[post]['distribution']]+'('+\
                        str(dataset.priors[post]['hyperparameters'][0])+','+\
                        str(dataset.priors[post]['hyperparameters'][1])+','+\
                        str(dataset.priors[post]['hyperparameters'][2])+','+\
                        str(dataset.priors[post]['hyperparameters'][3])
            s_param += ')$ & --- & \\textcolor{red}{add}'
        else:
            if dataset.priors[post]['distribution'] not in dists_types:
                print('BUG3')
                print(post)
                print(dataset.priors[post])
                print()
                quit()

            s_param += '$'+dists_types[dataset.priors[post]['distribution']]+'('+\
                        str(dataset.priors[post]['hyperparameters'][0])+','+\
                        str(dataset.priors[post]['hyperparameters'][1])
            if dataset.priors[post]['distribution'] == 'normal' or dataset.priors[post]['distribution']=='loguniform':
                s_param += '^2'

            s_param += ')$ &'

            print(priors_dict.keys())

            if post[:2] == 'GP':
                pvector = post.split('_')
                gp_param = pvector[1]
                insts = pvector[2:]
            if post in priors_dict.keys(): s_param += ' --- &' + priors_dict[post]['description']
            elif post[:2] == 'GP':
                pvector = post.split('_')
                gp_param = pvector[1]
                if 'GP_'+gp_param in priors_dict.keys(): 
                    s_param += ' --- &' + priors_dict['GP_'+gp_param]['description']
                else: 
                    s_param += ' --- & \\textcolor{red}{add}'
            else: s_param += ' --- & \\textcolor{red}{add}'

        s_param += linend
        fout.write(s_param+'\n')
        params_priors = np.delete(params_priors, np.where(params_priors == post)[0])


    if len(params_priors) > 1:
        print('% These params were not able to be latexifyed')
        print('%',params_priors) 
    tab_end = ['\\noalign{\\smallskip}', '\\hline', '\\hline', '\\end{tabular}', '\\end{table*}']
    for elem in tab_end:
        fout.write(elem+'\n')
    fout.close()
    return




















def print_posterior_table(dataset, results, precision=2, rvunits='ms'):
    # lin = 'parameter \t& ${}^{+{}}_{-{}}$ \\\\'.format(precision, precision, precision)
    # print(lin)

    out_folder = dataset.out_folder

    latex_fil = out_folder+'posterior_table.tex'
    print('writing to {}.'.format(latex_fil))
    fout = open(latex_fil, 'w')
    tab_start = ['\\begin{table*}', '\\centering', '\\caption{Posterior parameters}', '\\label{tab:posteriors}', '\\begin{tabular}{lc}',\
                '\\hline', '\\hline', '\\noalign{\\smallskip}', 'Parameter name & Posterior estimate \\\\',\
                '\\noalign{\\smallskip}', '\\hline', '\\hline']

    for elem in tab_start:
        fout.write(elem+'\n')

    linend = '\\\\[0.1 cm]'
    params_post = np.array([i for i in results.posteriors['posterior_samples'].keys()]) # transform it to array

    ## Stellar parameters first
    if 'rho' in params_post:
        fout.write('\\noalign{\\smallskip}\n')
        fout.write('Stellar Parameters '+linend+' \n')
        fout.write('\\noalign{\\smallskip}\n')
        s_param = '~~~'
        s_param += '$\\rho_{*} \, \mathrm{(kg/m^3)}$'
        s_param += '  & '

        val,valup,valdown = juliet.utils.get_quantiles(results.posteriors['posterior_samples']['rho'])
        errhi, errlo = round_sig(valup-val, sig=precision), round_sig(val-valdown, sig=precision)
        digits = np.max([np.abs(count_this(errhi)), np.abs(count_this(errlo))])
        vals = '{0:.{digits}f}'.format(val, digits = digits)
        errhis = '{0:.{digits}f}'.format(errhi, digits = digits)
        errlos = '{0:.{digits}f}'.format(errlo, digits = digits)
        s_param += '$'+vals+'^{+'+errhis+'}_{-'+errlos+'}$' + linend
        fout.write(s_param+'\n')
        params_post = np.delete(params_post, np.where(params_post == 'rho')[0])

    order_planetary = ['P', 't0', 'a', 'r1', 'r2', 'b', 'p', 'K', 'ecc', 'omega', 'sesinomega', 'secosomega',
                       'esinomega', 'ecosomega']
    planet_names = ['', 'b', 'c', 'd', 'e', 'f']
    # Save information stored in the prior: the dictionary, number of transiting planets, 
    # number of RV planets, numbering of transiting and rv planets (e.g., if p1 and p3 transit 
    # and all of them are RV planets, numbering_transit = [1,3] and numbering_rv = [1,2,3]). 
    # Save also number of *free* parameters (FIXED don't count here).

    rv_planets = dataset.numbering_rv_planets
    transiting_planets = dataset.numbering_transiting_planets
    for pl in np.unique(np.append(rv_planets, transiting_planets)):
        fout.write('\\noalign{\\smallskip}\n')
        if len(np.unique(np.append(rv_planets, transiting_planets))) == 1:
            fout.write('Planetary parameters'+linend+' \n')
        else:
            fout.write('Posterior parameters for planet {} '.format(planet_names[pl])+linend+' \n')
        fout.write('\\noalign{\\smallskip}\n')

        for param in order_planetary:
            key = '{}_p{}'.format(param, pl)

            if key not in params_post and key not in dataset.priors.keys():
                continue
            val,valup,valdown = juliet.utils.get_quantiles(results.posteriors['posterior_samples'][key])
            s_param = '~~~'
            if param == 't0': s_param += '$t_{0,' + planet_names[pl] + '}$'
            elif param == 'a': s_param += '$a_{' + planet_names[pl] + '}/R_*$'
            elif param == 'r1': s_param += '$r_{1,' + planet_names[pl] + '}$'
            elif param == 'r2': s_param += '$r_{2,' + planet_names[pl] + '}$'
            elif param == 'p': s_param += '$R_{' + planet_names[pl] + '}/R_*$'
            elif param == 'b': s_param += '$b = (a_{' + planet_names[pl] + '}/R_*) \\cos (i_{'+ planet_names[pl] +'}) $'
            elif param == 'ecc': s_param += '$e_{' + planet_names[pl] + '}$'
            elif param == 'omega': s_param += '$\\omega_{' + planet_names[pl] + '}$'
            elif 'sesinomega' in param:
                s_param += '$S_{1,' + planet_names[pl] + '} = \\sqrt{e_'+planet_names[pl]+'}\\sin \\omega_'+planet_names[pl]+'$'
            elif param == 'secosomega':
                s_param += '$S_{2,' + planet_names[pl] + '} = \\sqrt{e_'+planet_names[pl]+'}\\cos \\omega_'+planet_names[pl]+'$'
            elif param == 'esinomega':
                s_param += '$S_{1,' + planet_names[pl] + '} = e_'+planet_names[pl]+'\\sin \\omega_'+planet_names[pl]+'$'
            elif param == 'ecosomega':
                s_param += '$S_{1,' + planet_names[pl] + '} = e_'+planet_names[pl]+'\\cos \\omega_'+planet_names[pl]+'$'
            else: s_param += '$' + param + '_{' + planet_names[pl] + '}$'
            s_param += '  & '

            errhi, errlo = round_sig(valup-val, sig=precision), round_sig(val-valdown, sig=precision)
            # print(valup-val, errhi)
            # print(val-valdown, errlo)
            # print(count_this(errhi), count_this(errlo))
            digits = np.max([np.abs(count_this(errhi)), np.abs(count_this(errlo))])
            vals = '{0:.{digits}f}'.format(val, digits = digits)
            errhis = '{0:.{digits}f}'.format(errhi, digits = digits)
            errlos = '{0:.{digits}f}'.format(errlo, digits = digits)
            s_param += '$'+vals+'^{+'+errhis+'}_{-'+errlos+'}$' + linend 
            fout.write(s_param+'\n')
            params_post = np.delete(params_post, np.where(params_post == key)[0])


    instruments_rv = dataset.inames_rv
    instruments_lc = dataset.inames_lc

    if instruments_rv is not None and len(instruments_rv):
        orderinst_rv = ['mu','sigma_w']
        orderinst_rv_latex = ['\\mu', '\\sigma']

        fout.write('\\noalign{\\smallskip}\n')
        fout.write('RV instrumental parameters'+linend+'\n')
        fout.write('\\noalign{\\smallskip}\n')
        for inst in instruments_rv:
            for param,param_latex in zip(orderinst_rv,orderinst_rv_latex):
                s_param = '~~~'

                s_param += '$'+param_latex+'_{\\textnormal{'+inst+'}}$ (ms$^{-1}$)'
                # if param == 'mu':
                #     s_param += '$\\mu_{\\textnormal{' + inst + '}}$ (ms$^{-1}$)'
                # elif param == 'sigma_w':
                #     s_param += '$\\sigma_{\\textnormal{' + inst + '}}$ (ms$^{-1}$)'

                s_param += '  & '
                if '{}_{}'.format(param, inst) not in params_post:  # assume it was fixed
                    val = dataset.priors[param+'_'+inst]['hyperparameters']
                    s_param += '{} (fixed) {}'.format(val, linend)
                else:
                    val,valup,valdown = juliet.utils.get_quantiles(results.posteriors['posterior_samples']['{}_{}'.format(param, inst)])
                    errhi, errlo = round_sig(valup-val, sig=precision), round_sig(val-valdown, sig=precision)
                    # print(valup-val, val-valdown)
                    # print(errhi, errlo)
                    # print(count_this(errhi), count_this(errlo))
                    digits = np.max([np.abs(count_this(errhi)), np.abs(count_this(errlo))])
                    vals = '{0:.{digits}f}'.format(val, digits = digits)
                    errhis = '{0:.{digits}f}'.format(errhi, digits = digits)
                    errlos = '{0:.{digits}f}'.format(errlo, digits = digits)
                    s_param += '$'+vals+'^{+'+errhis+'}_{-'+errlos+'}$' + linend 
                
                fout.write(s_param+'\n')
                params_post = np.delete(params_post, np.where(params_post == '{}_{}'.format(param, inst))[0])

    if instruments_lc is not None and len(instruments_lc):
        orderinst_lc = ['mflux', 'sigma_w', 'q1', 'q2', 'theta0']#, 'mdilution']
        fout.write('\\noalign{\\smallskip}\n')
        fout.write('Photometry instrumental parameters'+linend+'\n')
        fout.write('\\noalign{\\smallskip}\n')
        for inst in instruments_lc:
            for param in orderinst_lc:
                s_param = '~~~'

                if param == 'mflux':
                    s_param += '$M_{\\textnormal{' + inst + '}}$ (ppm)'
                elif param == 'sigma_w':
                    s_param += '$\\sigma_{\\textnormal{' + inst + '}}$ (ppm)'
                elif param == 'q1':
                    s_param += '$q_{1,\\textnormal{' + inst + '}}$'
                elif param == 'q2' and '{}_{}'.format(param, inst) in params_post:
                    s_param += '$q_{2,\\textnormal{' + inst + '}}$'
                elif param == 'q2' and ('{}_{}'.format(param, inst) not in params_post or '{}_{}'.format(param, inst) not in dataset.priors.keys()):
                    continue
                
                # if param == 'theta' and ('{}0_{}'.format(param, inst) not in params_post or '{}0_{}'.format(param, inst) not in dataset.priors.keys()):
                    # continue
                # else:
                #     hastheta = True
                #     ind = 0
                #     while hastheta == True:
                elif param == 'theta0' and '{}_{}'.format(param, inst) in params_post:
                    s_param += '$\\theta_{0,\\textnormal{' + inst + '}}$'
                elif param == 'theta0' and ('{}_{}'.format(param, inst) not in params_post or '{}_{}'.format(param, inst) not in dataset.priors.keys()):
                    continue

                s_param += '  & '
                if '{}_{}'.format(param, inst) not in params_post:  # assume it was fixed
                    val = dataset.priors[param+'_'+inst]['hyperparameters']
                    s_param += '{} (fixed) {}'.format(val, linend)
                else:
                    val,valup,valdown = juliet.utils.get_quantiles(results.posteriors['posterior_samples']['{}_{}'.format(param, inst)])
                    errhi, errlo = round_sig(valup-val, sig=precision), round_sig(val-valdown, sig=precision)
                    # print(valup-val, val-valdown)
                    # print(errhi, errlo)
                    # print(count_this(errhi), count_this(errlo))
                    digits = np.max([np.abs(count_this(errhi)), np.abs(count_this(errlo))])
                    vals = '{0:.{digits}f}'.format(val, digits = digits)
                    errhis = '{0:.{digits}f}'.format(errhi, digits = digits)
                    errlos = '{0:.{digits}f}'.format(errlo, digits = digits)
                    s_param += '$'+vals+'^{+'+errhis+'}_{-'+errlos+'}$' + linend 
                
                fout.write(s_param+'\n')
                params_post = np.delete(params_post, np.where(params_post == '{}_{}'.format(param, inst))[0])

            fout.write('\\noalign{\\medskip}\n')

    gp_names = ['sigma', 'alpha', 'Gamma', 'Prot', 'B', 'L', 'C', 'timescale', 'rho', 'S0', 'Q', 'omega0']
    gp_names_latex = ['\\sigma', '\\alpha', '\\Gamma', 'P_{rot}', \
                    '\\textnormal{B}', '\\textnormal{L}', '\\textnormal{C}', '\\tau', \
                    '\\rho', 'S_0', '\\textnormal{Q}', '\\omega 0']

    if len(params_post) > 1:
        fout.write('\\noalign{\\smallskip}\n')
        fout.write('Additional parameters{}\n'.format(linend))
        fout.write('\\noalign{\\smallskip}\n')

    for post in params_post:
        s_param = '~~~'
        if post == 'unnamed': continue

        elif post == 'rho':
            s_param += '$\\rho_{*}$'
        elif post[:2] == 'GP':
            pvector = post.split('_')
            gp_param = pvector[1]
            insts = pvector[2:]

            for i,elem in enumerate(gp_names):
                if elem == gp_param: 
                    gp_param_latex = gp_names_latex[i]
                    continue

            if gp_param not in gp_names: # most likely alpha0, alpha1
                gp_param_latex = '\\alpha_{}'.format(gp_param[-1])

            s_param += gp_param_latex + '$^{GP}_{\\textnormal{'
            for inst in insts:
                s_param += inst + ','
            s_param = s_param[:-1]  # to get rid of the last comma
            s_param += '}}$'
        else:
            s_param += '$' + post + '$'
        s_param += '  & '

        val,valup,valdown = juliet.utils.get_quantiles(results.posteriors['posterior_samples']['{}'.format(post)])
        errhi, errlo = round_sig(valup-val, sig=precision), round_sig(val-valdown, sig=precision)
        # print(valup-val, val-valdown)
        # print(errhi, errlo)
        # print(count_this(errhi), count_this(errlo))
        digits = np.max([np.abs(count_this(errhi)), np.abs(count_this(errlo))])
        # print(digits)
        vals = '{0:.{digits}f}'.format(val, digits = digits)
        errhis = '{0:.{digits}f}'.format(errhi, digits = digits)
        errlos = '{0:.{digits}f}'.format(errlo, digits = digits)
        s_param += '$'+vals+'^{+'+errhis+'}_{-'+errlos+'}$' + linend 
        fout.write(s_param+'\n')
        params_post = np.delete(params_post, np.where(params_post == post)[0])

    if len(params_post) > 1:
        print('% These params were not able to be latexifyed')
        print('%',params_post) 
    tab_end = ['\\noalign{\\smallskip}', '\\hline', '\\hline', '\\end{tabular}', '\\end{table*}']
    for elem in tab_end:
        fout.write(elem+'\n')
    fout.close()
    # for post in posteriors.keys():

    #     print(post)

def round_sig(x, sig=2):
    y = round(x, sig-int(floor(log10(abs(x))))-1)
    if y >= 10: 
        return int(y)
    else: 
        return y

def count_this(x):
    d = decimal.Decimal(str(x))
    return d.as_tuple().exponent

def get_this(dist, sig=2):
    val,valup,valdown = juliet.utils.get_quantiles(dist)
    errhi, errlo = round_sig(valup-val, sig=sig), round_sig(val-valdown, sig=sig)
    # print(valup-val, val-valdown)
    # print(errhi, errlo)
    # print(count_this(errhi), count_this(errlo))
    digits = np.max([np.abs(count_this(errhi)), np.abs(count_this(errlo))])
    # print(digits)
    vals = '{0:.{digits}f}'.format(val, digits = digits)
    errhis = '{0:.{digits}f}'.format(errhi, digits = digits)
    errlos = '{0:.{digits}f}'.format(errlo, digits = digits)
    s_param += '$'+vals+'^{+'+errhis+'}_{-'+errlos+'}$' + linend 
    return s_param