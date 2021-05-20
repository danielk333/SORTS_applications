#!/usr/bin/env python
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from tqdm import tqdm

import sorts

#local
import config


def cost_function(Sigma_orb, num):
    
    stds = np.sqrt(np.diag(Sigma_orb))
    pos_err = np.mean(stds[:3])
    vel_err = np.mean(stds[3:])

    pos_cost = 1#per m
    vel_cost = 100#per m/s
    time_cost = 100#per s

    time_used = num*config.tracker.t_slice

    return pos_err*pos_cost + vel_err*vel_cost + time_used*time_cost


def test_scheduling_set():

    print('Scheduling for object:')
    print(str(config.obj) + '\n')

    print('First set of passes [all rx stations]')
    for ps in config.passes:
        print(ps)

    print('\n' + 'Setting num=10, frac=0.8')
    config.scheduler.set_observations(num=10, frac=0.8)

    print('\nObserve and determine orbit\n')
    Sigma_orb, data = config.scheduler.determine_orbit()

    print(f'Cost of observation: {cost_function(Sigma_orb,10)}\n')

    header = ['']+config.variables

    list_sig = (Sigma_orb).tolist()
    list_sig = [[var] + row for row,var in zip(list_sig, header[1:])]

    print(tabulate(list_sig, header, tablefmt="simple"))

    for ind, var in enumerate(config.variables):
        if ind > 2:
            unit = 'km/s'
        else:
            unit = 'km'
        print(f'{var}-std: {np.sqrt(Sigma_orb[ind,ind])*1e-3:.2f} {unit}')

    fig, ax = sorts.plotting.observed_parameters(
        [data[0]],
        passes=[config.passes[0]], 
        snrdb_lim = 10.0
    )

    plt.show()


def get_param_cost(num, frac):
    config.scheduler.set_observations(num=num, frac=frac)
    Sigma_orb, _ = config.scheduler.determine_orbit()
    return cost_function(Sigma_orb,num)

def grid_search_cost_function(n_steps, f_steps):

    max_n = int(config.t_pass/config.tracker.t_slice)

    N, F = np.meshgrid(
        n_steps,
        f_steps, 
        sparse=False, 
        indexing='ij',
    )

    cost = np.zeros_like(F)

    p = tqdm(total=np.prod(N.shape))

    for i in range(N.shape[0]):
        for j in range(N.shape[1]):
            cost[i,j] = get_param_cost(N[i,j], F[i,j])
            p.update(1)

    p.close()

    return N, F, cost


if __name__=='__main__':
    #test_scheduling_set()

    n_steps = np.array(list(range(1,10)) + [20])
    f_steps = np.linspace(0.1,1,num=10)

    if pathlib.Path('N.npy').is_file():
        N = np.load('N.npy')
        F = np.load('F.npy')
        cost = np.load('cost.npy')
    else:
        N, F, cost = grid_search_cost_function(n_steps, f_steps)
        np.save('N.npy', N)
        np.save('F.npy', F)
        np.save('cost.npy', cost)

    
    fig, ax = plt.subplots()
    c = ax.pcolormesh(N[1:,:], F[1:,:], cost[1:,:])
    xtickslocs = ax.get_xticks()
    ytickslocs = ax.get_yticks()
    ax.set_xticks(xtickslocs)
    ax.set_xticklabels(['{x}' for x in n_steps[1:]])
    ax.set_title('Cost function')
    fig.colorbar(c, ax=ax)

    fig, ax = plt.subplots()
    N_ = N[1:,:]
    N_[-1,:] = 11
    c = ax.pcolormesh(N_, F[1:,:], np.log10(cost[1:,:]))

    ax.set_xticks([x + 0.5 for x in n_steps[1:-1]] + [11])
    ax.set_xticklabels([f'{x}' for x in n_steps[1:]])

    ax.set_title('Log-Cost function', fontsize=16)
    ax.set_xlabel('Number of measurement points', fontsize=16)
    ax.set_ylabel('Fraction of pass tracked', fontsize=16)
    cb = fig.colorbar(c, ax=ax)
    cb.ax.set_ylabel('log10(COST)', fontsize=16)

    plt.show()
