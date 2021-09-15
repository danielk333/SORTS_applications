#!/usr/bin/env python
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from tqdm import tqdm

import sorts

#local
import scheduler

np.random.seed(2314)

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

    def optim_func(params):
        spent = ts.spend_pulses(ts.pulses - ts.base_pulses, params)
        print(f'spent pulses {spent}')
        nums = ts.base_nums + spent
        nums = nums.astype(np.int32)
        ts.set_schedule(nums)
        Sigma_orbs = ts.determine_orbits()

        pos_cost = 1#per m
        vel_cost = 100#per m/s

        cost = 0
        pos_vel_errs = []
        for sig, pri in zip(Sigma_orbs, ts.priority):

            stds = np.sqrt(np.diag(sig))
            pos_err = np.mean(stds[:3])
            vel_err = np.mean(stds[3:])

            pos_vel_errs.append((pos_err, vel_err))

            cost += (pos_err*pos_cost + vel_err*vel_cost)*pri

        return cost, spent, pos_vel_errs

    #pulse spending function is 
    # distribution = np.log((self.priority + params[1])/params[0])
    # hence min(params[1]) > -min(self.priority)

    scale = 10.0**np.linspace(-2,2,num=10)
    offset = np.linspace(0, np.max(ts.priority)*0.2, num=3)
    PX, PY = np.meshgrid(
        scale,
        offset, 
        sparse=False, 
        indexing='ij',
    )



    if pathlib.Path('cost.npy').is_file():
        cost = np.load('cost.npy')
    else:
        p = tqdm(total=np.prod(PX.shape))

        cost = np.zeros_like(PX)
        for i in range(PX.shape[0]):
            for j in range(PX.shape[1]):
                cost[i,j], spent, pos_vel_errs = optim_func([PX[i,j], PY[i,j]])
                p.update(1)

        p.close()

        np.save('cost.npy', cost)


    #Other ideas:
    # Just create baseline for objects of a certain priority
    # Then see what functions can bring more objects over the threshold while still improving important objects
    # See how different algorithms behave in different restriction environments
    # Also check what distribution functions are good
    # Try to directly optimize distribution function values instead
    # Different cost functions
    # Different stages of caching results and optimization
    # How to introduce errors in SN due to catalog errors in covariance estimation: random sampling? distribution proagation? scaling?
    # ...
    # ...

    
    fig, ax = plt.subplots()
    c = ax.pcolormesh(PX, PY, np.log10(cost))

    ax.set_title('Log-Cost function', fontsize=16)
    ax.set_xlabel('Scale', fontsize=16)
    ax.set_ylabel('Offset', fontsize=16)
    cb = fig.colorbar(c, ax=ax)
    cb.ax.set_ylabel('log10(COST)', fontsize=16)

    fig, ax = plt.subplots()
    c = ax.pcolormesh(np.log10(PX), PY, np.log10(cost))

    ax.set_title('Log-Cost function', fontsize=16)
    ax.set_xlabel('Scale', fontsize=16)
    ax.set_ylabel('Offset', fontsize=16)
    cb = fig.colorbar(c, ax=ax)
    cb.ax.set_ylabel('log10(COST)', fontsize=16)


    plt.show()


    params = [
        (0.1,0),
        (10,0),
        (1,10),
        (10,10),
    ]

    fig, ax = plt.subplots(1,1)

    for param in params:

        nums_ = ts.spend_pulses(ts.pulses - ts.base_pulses, param, x=np.arange(10)).astype(np.float64)/(ts.pulses - ts.base_pulses)
        ax.plot(np.arange(10), nums_, label=f'a = {param[0]}, b = {param[1]}')

        cost, spent, pos_vel_errs = optim_func(param)
        print(f'Cost: {cost} @ a = {param[0]}, b = {param[1]}')
        
        print(f'Base spent        : {ts.base_nums}')
        print(f'Pulses spent      : {spent + ts.base_nums}')
        print(f'Pulses distributed: {spent}')

        for oid, errs in enumerate(pos_vel_errs):
            print(f'Object {oid}: {errs[0]*1e-3} km | {errs[1]*1e-3} km/s')

    ax.set_ylabel('Fraction of pulses distributed')
    ax.set_xlabel('Priority number')
    ax.legend()
    plt.show()


