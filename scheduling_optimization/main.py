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
    epoch = pop.get_object(0).epoch
    print(f'Population size: {len(pop)}')

    #THE IDEA IS:
    # - start with the baseline
    # - use function to start adding measurement points to objects

    ts.setup_scheduler(
        population = pop, 
        t_start = 0.0, 
        t_end = 24*3600.0, 
        priority = np.arange(len(pop)), 
        pulses = 100, 
        min_Sigma_orb = [2e3, 2e3], 
        epoch = epoch, 
        t_restriction = None,
    )

    #ts.set_base_observations()
    ts.base_nums = np.array([1,1,18])
    ts.base_pulses = 20

    print(f'Object base tracklet points {ts.base_nums}')
    print(f'Pulses used for base {ts.base_pulses}, pulses left {ts.pulses - ts.base_pulses}')

    def optim_func(scaling):
        nums = ts.base_nums + ts.spend_pulses(ts.pulses - ts.base_pulses, scaling)
        nums = nums.astype(np.int32)
        ts.set_schedule(nums)
        Sigma_orbs = ts.determine_orbits()

        pos_cost = 1#per m
        vel_cost = 100#per m/s

        cost = 0
        for sig, pri in zip(Sigma_orbs, ts.priority):

            stds = np.sqrt(np.diag(sig))
            pos_err = np.mean(stds[:3])
            vel_err = np.mean(stds[3:])

            cost += (pos_err*pos_cost + vel_err*vel_cost)*pri

        return cost


    print([optim_func(x) for x in np.linspace(0,1,5)])


