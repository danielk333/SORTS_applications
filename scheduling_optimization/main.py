#!/usr/bin/env python
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from tqdm import tqdm

import sorts

#local
import scheduler



if __name__=='__main__':
    print('Init scheduler')
    ts = scheduler.TrackingScheduler(
        radar = sorts.radars.eiscat3d,
        error_model_cache = pathlib.Path('/home/danielk/IRF/IRF_GITLAB/SORTS/examples/data/'),
    )

    print('Loading population')
    pop = sorts.Population.load(
        pathlib.Path('../scan_and_chase/sample_population.h5'),     
        propagator = sorts.propagator.SGP4,
        propagator_options = dict(
            settings = dict(
                in_frame='TEME',
                out_frame='ITRS',
            ),
        ),
    )

    pop.data = pop.data[:3]
    print(f'Population size: {len(pop)}')

    ts.setup_scheduler(
        population = pop, 
        t_start = 0.0, 
        t_end = 12*3600.0, 
        priority = np.arange(len(pop)), 
        pulses = 100, 
        min_Sigma_orb=None, 
        epoch=None, 
        t_restriction=None,
    )

    ts.set_base_observations()


