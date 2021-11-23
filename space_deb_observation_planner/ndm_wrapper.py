import subprocess
import shutil
import logging

import numpy as np
from astropy.time import TimeDelta

import sorts
import pyorb

import configuration as cfg

logger = logging.getLogger('sst-cli.ndm_wrapper')


def check_setup():
    bin_file = cfg.CACHE / 'nasa-breakup-model' / 'bin' / 'breakup'
    nbm_check = bin_file.is_file()
    logger.info(f'nasa-breakup-model = {nbm_check}')
    return nbm_check


def setup():
    '''Setup requrements for this CLI
    '''

    get_cmd = ['git', 'clone', 'https://gitlab.obspm.fr/apetit/nasa-breakup-model.git']

    if not cfg.MODEL_DIR.is_dir():
        subprocess.check_call(get_cmd, cwd=str(cfg.MODEL_DIR))
    
    subprocess.check_call(['make'], cwd=str(cfg.MODEL_DIR / 'src'))


def run_nasa_breakup(nbm_param_file, nbm_param_type, space_object, fragmentation_time):
    old_inps = (cfg.MODEL_DIR / 'bin' / 'inputs').glob('nbm_param_[0-9]*.txt')
    for file in old_inps:
        print(f'Removing old input file {file}')
        file.unlink()

    shutil.copy(nbm_param_file, str(cfg.MODEL_DIR / 'bin' / 'inputs' / 'nbm_param_1.txt'))

    event_time = space_object.epoch + TimeDelta(fragmentation_time*3600.0, format='sec')

    state_itrs = space_object.get_state([fragmentation_time*3600.0])
    state_teme = sorts.frames.convert(
        event_time, 
        state_itrs, 
        in_frame='ITRS', 
        out_frame='TEME',
    )
    orbit_teme = pyorb.Orbit(
        cartesian=state_teme,
        M0 = pyorb.M_earth,
        m = space_object.parameters.get('m', 0.0),
        degrees=True,
        type='mean',
    )
    kepler_teme = orbit_teme.kepler.flatten()

    sched_header = [
        'ID_NORAD', 'J_DAY', 'YYYY', 
        'MM', 'DD', 'NAME', 
        'TYPE', 'SMA[m]', 'ECC', 
        'INC[deg]', 'RAAN[deg]', 'A_PER[deg]', 
        'MA[deg]', 'MASS[kg]', 'S',
    ]
    sched_data = [
        '00000001', 
        f'{event_time.jd:.6f}', 
        str(event_time.datetime.year), 
        str(event_time.datetime.month), 
        str(event_time.datetime.day),
        str(space_object.oid),
        nbm_param_type,
        f'{kepler_teme[0]:.8e}',
        f'{kepler_teme[1]:.8e}',
        f'{kepler_teme[2]:.8e}',
        f'{kepler_teme[4]:.8e}',
        f'{kepler_teme[3]:.8e}',
        f'{kepler_teme[5]:.8e}',
        str(space_object.parameters.get('m', 0.0)),
        str(1.0),  # scale factor
    ]

    max_key_len = max([len(x) for x in sched_header])
    print('Fragmentation model with object:')
    for key, val in zip(sched_header, sched_data):
        print(f'{key:<{max_key_len}}: {val}')

    sched_file = cfg.MODEL_DIR / 'bin' / 'inputs' / 'schedule.txt'
    print(f'Writing schedule file: {sched_file}')
    with open(sched_file, 'w') as fh:
        fh.write(' '.join(sched_header) + '\n')
        fh.write(' '.join(sched_data))

    print('Running nasa-breakup-model...')
    subprocess.check_call(['./breakup'], cwd=str(cfg.MODEL_DIR / 'bin'))
    print('nasa-breakup-model done')
    
    data_file = cfg.MODEL_DIR / 'bin' / 'outputs' / 'cloud_cart.txt'

    cloud_data = np.genfromtxt(
        str(data_file),
        skip_header=1,
    )
    with open(data_file, 'r') as fh:
        header = fh.readline().split()

    return cloud_data, header
