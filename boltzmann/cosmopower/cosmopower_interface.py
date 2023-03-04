from builtins import str
import os
from cosmosis.datablock import names, option_section
import sys
import traceback
import warnings
import scipy
# add directory to the path
dirname = os.path.split(__file__)[0]
# enable debugging from the same directory
if not dirname.strip():
    dirname = '.'

import numpy as np

# These are pre-defined strings we use as datablock
# section names
cosmo = names.cosmological_parameters
distances = names.distances
cmb_cl = names.cmb_cl


def setup(options):


    import cosmopower as cp
    from cosmopower import cosmopower_NN
    from cosmopower import cosmopower_PCAplusNN


    derived_params_names = ['100*theta_s',
                            'sigma8',
                            'YHe',
                            'z_reio',
                            'Neff',
                            'tau_rec',
                            'z_rec',
                            'rs_rec',
                            'ra_rec',
                            'tau_star',
                            'z_star',
                            'rs_star',
                            'ra_star',
                            'rs_drag']


    emulator_dict = {}
    emulator_dict['lcdm'] = {}
    emulator_dict['mnu'] = {}
    emulator_dict['neff'] = {}
    emulator_dict['wcdm'] = {}




    emulator_dict['lcdm']['TT'] = 'TT_v1'
    emulator_dict['lcdm']['TE'] = 'TE_v1'
    emulator_dict['lcdm']['EE'] = 'EE_v1'
    emulator_dict['lcdm']['PP'] = 'PP_v1'
    emulator_dict['lcdm']['PKNL'] = 'PKNL_v1'
    emulator_dict['lcdm']['PKL'] = 'PKL_v1'
    emulator_dict['lcdm']['DER'] = 'DER_v1'
    emulator_dict['lcdm']['DAZ'] = 'DAZ_v1'
    emulator_dict['lcdm']['HZ'] = 'HZ_v1'
    emulator_dict['lcdm']['S8Z'] = 'S8Z_v1'

    emulator_dict['mnu']['TT'] = 'TT_mnu_v1'
    emulator_dict['mnu']['TE'] = 'TE_mnu_v1'
    emulator_dict['mnu']['EE'] = 'EE_mnu_v1'
    emulator_dict['mnu']['PP'] = 'PP_mnu_v1'
    emulator_dict['mnu']['PKNL'] = 'PKNL_mnu_v1'
    emulator_dict['mnu']['PKL'] = 'PKL_mnu_v1'
    emulator_dict['mnu']['DER'] = 'DER_mnu_v1'
    emulator_dict['mnu']['DAZ'] = 'DAZ_mnu_v1'
    emulator_dict['mnu']['HZ'] = 'HZ_mnu_v1'
    emulator_dict['mnu']['S8Z'] = 'S8Z_mnu_v1'


    emulator_dict['neff']['TT'] = 'TT_neff_v1'
    emulator_dict['neff']['TE'] = 'TE_neff_v1'
    emulator_dict['neff']['EE'] = 'EE_neff_v1'
    emulator_dict['neff']['PP'] = 'PP_neff_v1'
    emulator_dict['neff']['PKNL'] = 'PKNL_neff_v1'
    emulator_dict['neff']['PKL'] = 'PKL_neff_v1'
    emulator_dict['neff']['DER'] = 'DER_neff_v1'
    emulator_dict['neff']['DAZ'] = 'DAZ_neff_v1'
    emulator_dict['neff']['HZ'] = 'HZ_neff_v1'
    emulator_dict['neff']['S8Z'] = 'S8Z_neff_v1'



    emulator_dict['wcdm']['TT'] = 'TT_w_v1'
    emulator_dict['wcdm']['TE'] = 'TE_w_v1'
    emulator_dict['wcdm']['EE'] = 'EE_w_v1'
    emulator_dict['wcdm']['PP'] = 'PP_w_v1'
    emulator_dict['wcdm']['PKNL'] = 'PKNL_w_v1'
    emulator_dict['wcdm']['PKL'] = 'PKL_w_v1'
    emulator_dict['wcdm']['DER'] = 'DER_w_v1'
    emulator_dict['wcdm']['DAZ'] = 'DAZ_w_v1'
    emulator_dict['wcdm']['HZ'] = 'HZ_w_v1'
    emulator_dict['wcdm']['S8Z'] = 'S8Z_w_v1'


    cp_tt_nn = {}
    cp_te_nn = {}
    cp_ee_nn = {}
    cp_pp_nn = {}
    cp_pknl_nn = {}
    cp_pkl_nn = {}
    cp_der_nn = {}
    cp_da_nn = {}
    cp_h_nn = {}
    cp_s8_nn = {}
    path_to_cosmopower_organization = '/Users/boris/Work/CLASS-SZ/SO-SZ/cosmopower-organization/'

    for mp in ['lcdm','mnu','neff','wcdm']:
        path_to_emulators = path_to_cosmopower_organization + mp +'/'

        cp_tt_nn[mp] = cosmopower_NN(restore=True,
                                 restore_filename=path_to_emulators + 'TTTEEE/' + emulator_dict[mp]['TT'])

        cp_te_nn[mp] = cosmopower_PCAplusNN(restore=True,
                                        restore_filename=path_to_emulators + 'TTTEEE/' + emulator_dict[mp]['TE'])

        cp_ee_nn[mp] = cosmopower_NN(restore=True,
                                 restore_filename=path_to_emulators + 'TTTEEE/' + emulator_dict[mp]['EE'])

        cp_pp_nn[mp] = cosmopower_NN(restore=True,
                                 restore_filename=path_to_emulators + 'PP/' + emulator_dict[mp]['PP'])

        cp_pknl_nn[mp] = cosmopower_NN(restore=True,
                                   restore_filename=path_to_emulators + 'PK/' + emulator_dict[mp]['PKNL'])

        cp_pkl_nn[mp] = cosmopower_NN(restore=True,
                                  restore_filename=path_to_emulators + 'PK/' + emulator_dict[mp]['PKL'])

        cp_der_nn[mp] = cosmopower_NN(restore=True,
                                  restore_filename=path_to_emulators + 'derived-parameters/' + emulator_dict[mp]['DER'])

        cp_da_nn[mp] = cosmopower_NN(restore=True,
                                 restore_filename=path_to_emulators + 'growth-and-distances/' + emulator_dict[mp]['DAZ'])

        cp_h_nn[mp] = cosmopower_NN(restore=True,
                                restore_filename=path_to_emulators + 'growth-and-distances/' + emulator_dict[mp]['HZ'])

        cp_s8_nn[mp] = cosmopower_NN(restore=True,
                                 restore_filename=path_to_emulators + 'growth-and-distances/' + emulator_dict[mp]['S8Z'])


    # Read options from the ini file which are fixed across
    # the length of the chain
    config = {
        'lmax': options.get_int(option_section, 'lmax', default=2000),
        'zmax': options.get_double(option_section, 'zmax', default=3.0),
        'kmax': 50.,#options.get_double(option_section, 'kmax', default=50.0),
        'debug': options.get_bool(option_section, 'debug', default=False),
        'lensing': options.get_bool(option_section, 'lensing', default=True),
        'cmb': options.get_bool(option_section, 'cmb', default=True),
        'mpk': options.get_bool(option_section, 'mpk', default=True),
        'save_matter_power_lin': options.get_bool(option_section, 'save_matter_power_lin', default=True),
        'save_cdm_baryon_power_lin': options.get_bool(option_section, 'save_cdm_baryon_power_lin', default=False),
    }


    # for _, key in options.keys(option_section):
    #     if key.startswith('class_'):
    #         config[key] = options[option_section, key]


    # Create the object that connects to Class
    # config['cosmo'] = classy.Class()


    config['pkl_emu'] = cp_pkl_nn
    config['pknl_emu'] = cp_pknl_nn
    config['s8_emu'] = cp_s8_nn
    config['h_emu'] = cp_h_nn
    config['da_emu'] = cp_da_nn
    config['der_emu'] = cp_der_nn


    # Return all this config information
    return config

def choose_outputs(config):
    outputs = []
    if config['cmb']:
        outputs.append("tCl pCl")
    if config['lensing']:
        outputs.append("lCl")
    if config["mpk"]:
        outputs.append("mPk")
    return " ".join(outputs)

def get_class_inputs(block, config):

    # Get parameters from block and give them the
    # names and form that class expects
    nnu = block.get_double(cosmo, 'nnu', 3.046)
    nmassive = block.get_int(cosmo, 'num_massive_neutrinos', default=0)


    params = {
        'output': choose_outputs(config),
        'lensing':   'yes' if config['lensing'] else 'no',
        'A_s':       block[cosmo, 'A_s'],
        'n_s':       block[cosmo, 'n_s'],
        'H0':        100 * block[cosmo, 'h0'],
        'omega_b':   block[cosmo, 'ombh2'],
        'omega_cdm': block[cosmo, 'omch2'],
        'tau_reio':  block[cosmo, 'tau'],
        'T_cmb':     block.get_double(cosmo, 'TCMB', default=2.726),
        # 'N_ur':      nnu - nmassive,
        # 'N_ncdm':    nmassive,
        # 'm_ncdm':    block.get_double(cosmo, 'mnu', default=0.06)

     # CLASS SETTINGS FOR COSMOPOWER
      'N_ncdm': 1,
      'N_ur': 2.0328,
      'm_ncdm': 0.06


    }

    if config["cmb"] or config["lensing"]:
        params.update({
          'l_max_scalars': config["lmax"],
        })


    if config["mpk"]:
        params.update({
            'P_k_max_h/Mpc':  config["kmax"],
            'z_pk': ', '.join(str(z) for z in np.arange(0.0, config['zmax'], 0.01)),
            'z_max_pk': config['zmax'],
        })



    if block.has_value(cosmo, "massless_nu"):
        warnings.warn("Parameter massless_nu is being ignored. Set nnu, the effective number of relativistic species in the early Universe.")

    if (block.has_value(cosmo, "omega_nu") or block.has_value(cosmo, "omnuh2")) and not (block.has_value(cosmo, "mnu")):
        warnings.warn("Parameter omega_nu and omnuh2 are being ignored. Set mnu and num_massive_neutrinos instead.")


    # for key,val in config.items():
    #     if key.startswith('class_'):
    #         params[key[6:]] = val


    return params




def get_class_outputs(block, config):
    ##
    # Derived cosmological parameters
    ##

    h0 = block[cosmo, 'h0']

    ##
    # Matter power spectrum
    ##

    # Ranges of the redshift and matter power
    nk = 5000 # do not touch this (it belongs to the emulators architecture)
    ndspl = 10 # do not touch this (it belongs to the emulators architecture)
    ndspl_fastpt = 5
    # Define k,z we want to sample
    zmax = 2.
    nz = 50 # anything between 15 and 500... this is determining for time.
    z = np.linspace(0.,zmax,nz)
    k = np.geomspace(1e-4,50.,nk)[::ndspl] # (k-range of emulators is 1e-4, 50)
    ls = np.arange(2,5000+2)[::ndspl] # (sorry, dont care about this, but leave it there; scaling to p(k))
    dls = ls*(ls+1.)/2./np.pi # (sorry, dont care about this, but leave it there; scaling to p(k))
    nz = len(z)
    nk = len(k)


    # setting up the paramter dictionnary that we feed to the emulator
    params = get_class_inputs(block, config)
    params_dict = {# LambdaCDM parameters
                   # last column of Table 1 of https://arxiv.org/pdf/1807.06209.pdf:
                   'H0': [params['H0']],
                   'omega_b': [params['omega_b']],
                   'omega_cdm': [params['omega_cdm']],
                   'ln10^{10}A_s': [np.log(1e10*params['A_s'])],
                   'tau_reio': [params['tau_reio']],
                   'n_s': [params['n_s']],
                   }

    # cosmopower call, the derived parameter emulator, e.g. rs_drag and other things
    der_params = config['der_emu']['lcdm'].ten_to_predictions_np(params_dict.copy())

    # for emulators of distances and growth:
    nz_arr = 5000 # number of z-points in redshift data (dont touch this)
    z_arr_max = 20. # max redshift of redshift data (dont touch this)
    z_arr = np.linspace(0.,z_arr_max,nz_arr)#  (dont touch this)

    # Extract (interpolate) P(k,z) at the requested
    # sample points.
    # if 'mPk' in c.pars['output']:
    if 'mPk' in 'mPk':    # for now always do:
        params_dict_pp = params_dict.copy()
            # params_cp[key] = [value]
        # get sigma8(z), useful to fsigma8 computation, and also f, the growth rate. 
        predicted_s8z = config['s8_emu']['lcdm'].predictions_np(params_dict_pp)
        s8z_interp = scipy.interpolate.interp1d(
                                        z_arr,
                                        predicted_s8z[0],
                                        kind='linear',
                                        axis=-1,
                                        copy=True,
                                        bounds_error=False,
                                        fill_value=np.nan,
                                        assume_sorted=False)
        # get sigma8 at z=0:
        block[cosmo, 'sigma_8'] = der_params[0][1]

        # Total matter power spectrum (saved as grid)
        if config['save_matter_power_lin']:
            P = np.zeros((k[::ndspl_fastpt].size, z.size))
            for j, zi in enumerate(z):
                params_dict_pp = params_dict.copy()
                params_dict_pp.pop('tau_reio')
                params_dict_pp['z_pk_save_nonclass'] = [zi]
                respk = config['pkl_emu']['lcdm'].predictions_np(params_dict_pp)[0]
                respk = ((dls)**-1*10.**np.asarray(respk))
                P[:,j] = respk[::ndspl_fastpt]
            # print(np.shape(P * h0**3))
            # exit(0)
            block.put_grid("matter_power_lin", "k_h", k[::ndspl_fastpt] / h0, "z", z, "P_k", P * h0**3)
            # block.put_grid("matter_power_lin", "k_h", k , "z", z, "P_k", P * h0**0)

        # CDM+baryons power spectrum
        if config['save_cdm_baryon_power_lin']:
            # P = np.zeros((k.size, z.size))
            # for i, ki in enumerate(k):
            #     for j, zi in enumerate(z):
            #         P[i, j] = c.pk_cb_lin(ki, zi)
            # block.put_grid('cdm_baryon_power_lin', 'k_h', k/h0, 'z', z, 'P_k', P*h0**3)\
            print('save_cdm_baryon_power_lin not available with cosmopower yet')
            exit(0)


        # Get growth rates and sigma_8
        # params_cp = {}



        def effective_f_sigma8(zf):
            """
            effective_f_sigma8(z)

            Returns the time derivative of sigma8(z) computed as (d sigma8/d ln a)

            Parameters
            ----------
            z : float
                    Desired redshift
            z_step : float
                    Default step used for the numerical two-sided derivative. For z < z_step the step is reduced progressively down to z_step/10 while sticking to a double-sided derivative. For z< z_step/10 a single-sided derivative is used instead.

            Returns
            -------
            (d ln sigma8/d ln a)(z) (dimensionless)
            """
            # we need d sigma8/d ln a = - (d sigma8/dz)*(1+z)

            # if possible, use two-sided derivative with default value of z_step
            z_step=0.1
            if zf >= z_step:
                result = (s8z_interp(zf-z_step)-s8z_interp(zf+z_step))/(2.*z_step)*(1+zf)
            else:
                # if z is between z_step/10 and z_step, reduce z_step to z, and then stick to two-sided derivative
                if (zf > z_step/10.):
                    z_step = zf
                    result = (s8z_interp(zf-z_step)-s8z_interp(zf+z_step))/(2.*z_step)*(1+zf)
                else:
                    z_step /=10
                    result = (s8z_interp(zf)-s8z_interp(zf+z_step))/z_step*(1+zf)
            return result



        def scale_independent_growth_factor_f(zf):
            return effective_f_sigma8(zf)/s8z_interp(zf)

        def scale_independent_growth_factor(zf):
            return s8z_interp(zf)/s8z_interp(0.)

        effective_f_sigma8 = np.vectorize(effective_f_sigma8)
        scale_independent_growth_factor_f = np.vectorize(scale_independent_growth_factor_f)
        scale_independent_growth_factor = np.vectorize(scale_independent_growth_factor)


        D = scale_independent_growth_factor(z)
        f = scale_independent_growth_factor_f(z)
        fsigma = effective_f_sigma8(z)
        sigma_8_z = s8z_interp(z)


        block[names.growth_parameters, "z"] = z
        block[names.growth_parameters, "sigma_8"] = np.array(sigma_8_z)
        block[names.growth_parameters, "fsigma_8"] = np.array(fsigma)
        block[names.growth_parameters, "d_z"] = np.array(D)
        block[names.growth_parameters, "f_z"] = np.array(f)
        block[names.growth_parameters, "a"] = 1/(1+z)

        # if c.nonlinear_method != 0:
        if 1 != 0: # for now always do:
            P = np.zeros((k[::ndspl_fastpt].size, z.size))
            for j, zi in enumerate(z):
                params_dict_pp = params_dict.copy()
                params_dict_pp.pop('tau_reio')
                params_dict_pp['z_pk_save_nonclass'] = [zi]
                respk = config['pknl_emu']['lcdm'].predictions_np(params_dict_pp)[0]
                respk = ((dls)**-1*10.**np.asarray(respk))
                P[:,j] = respk[::ndspl_fastpt]

            block.put_grid("matter_power_nl", "k_h", k[::ndspl_fastpt] / h0, "z", z, "P_k", P * h0**3)


    ##
    # Distances and related quantities
    ##

    # save redshifts of samples
    nz = 250 # anything between 15 and 500... this is determining for time.
    z = np.linspace(0.,2.,nz)
    block[distances, 'z'] = z
    block[distances, 'nz'] = nz
    block[distances, 'a'] = 1/(1+z)
    # Save distance samples


    params_dict_pp = params_dict.copy()
    predicted_daz = config['da_emu']['lcdm'].predictions_np(params_dict_pp)
    daz_interp = scipy.interpolate.interp1d(
                                    z_arr,
                                    predicted_daz[0],
                                    kind='linear',
                                    axis=-1,
                                    copy=True,
                                    bounds_error=False,
                                    fill_value='extrapolate',
                                    assume_sorted=False)



    # d_a = np.array([c.angular_distance(zi) for zi in z])

    d_a = daz_interp(z)



    block[distances, 'd_a'] = d_a
    block[distances, 'd_l'] = d_a * (1 + z)**2
    block[distances, 'd_m'] = d_a * (1 + z)

    # Save some auxiliary related parameters
    # block[distances, 'age'] = c.age()
    block[distances, 'rs_zdrag'] = der_params[0][13]

    ##
    # Now the CMB C_ell
    ##
    if config["cmb"]:
        print('cmb not implemented yet')
        exit(0)
        # c_ell_data = c.lensed_cl() if config['lensing'] else c.raw_cl()
        # ell = c_ell_data['ell']
        # ell = ell[2:]
        #
        # # Save the ell range
        # block[cmb_cl, "ell"] = ell
        #
        # # t_cmb is in K, convert to mu_K, and add ell(ell+1) factor
        # tcmb_muk = block[cosmo, 'tcmb'] * 1e6
        # f = ell * (ell + 1.0) / 2 / np.pi * tcmb_muk**2
        #
        # # Save each of the four spectra
        # for s in ['tt', 'ee', 'te', 'bb']:
        #     block[cmb_cl, s] = c_ell_data[s][2:] * f


def execute(block, config):
    get_class_outputs(block, config)
    return 0


def cleanup(config):
    return 0
