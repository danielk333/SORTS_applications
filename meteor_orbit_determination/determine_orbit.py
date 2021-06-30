#!/usr/bin/env python

'''
Calculating distributions pre-encounter orbits
=================================================

'''

import pathlib
import datetime

import numpy as np
from numpy.random import default_rng
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from astropy.time import Time, TimeDelta

import sorts
import pyorb

# pth = pathlib.Path('/home/danielk/IRF/data/high_alt/check/20100812T103919310_IPP455_512/20100812T103919310_IPP455_512_states.h5')
pth = pathlib.Path('/home/danielk/IRF/data/high_alt/check/20100812T034755280_IPP457_569/20100812T034755280_IPP457_569_states.h5')
kernel = '/home/danielk/IRF/IRF_GITLAB/EPHEMERIS_FILES/de430.bsp'
samples = 5000

def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.

    Source: https://gist.github.com/victorkristof/b9d794fe1ed12e708b9d
    
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    return datetime.datetime.fromordinal(int(datenum)) \
           + datetime.timedelta(days=days) \
           - datetime.timedelta(days=366)


with h5py.File(pth, 'r') as h:
    #saved transposed
    states = h['sample_states'][()].T
    
    #thinning
    rng = default_rng()
    inds = rng.choice(states.shape[1], size=samples, replace=False)
    states = states[:,inds]

    num = states.shape[1]
    datetime_epoch = datenum_to_datetime(h.attrs['epoch'][0])
    epoch = Time(datetime_epoch, scale='utc', format='datetime')

states_mu = np.mean(states,axis=1)

print(f'Encounter Geocentric range (ENU): {np.linalg.norm(states_mu[:3])*1e-3} km')
print(f'Encounter Geocentric speed (ENU): {np.linalg.norm(states_mu[3:])*1e-3} km/s')
print(f'At epoch: {epoch.iso}')
print(f'Samples: {num}')
print(f'Using JPL kernel: {kernel}')

lat = (34.0 + (51.0 + 14.50/60.0)/60.0 );
lon = (136.0 + (06.0 + 20.24/60.0)/60.0 );
alt = 372.0;

radar_itrs = sorts.frames.geodetic_to_ITRS(lat, lon, alt)
states_ITRS = states.copy()
states_ITRS[:3,:] = radar_itrs[:,None] + sorts.frames.enu_to_ecef(lat, lon, alt, states[:3,:])
states_ITRS[3:,:] = sorts.frames.enu_to_ecef(lat, lon, alt, states[3:,:])


def propagate_pre_encounter(state, epoch, in_frame, out_frame, kernel, dt = 10.0, t_target = -7*3600.0, settings = None):

    t = np.array([t_target])

    reb_settings = dict(
        in_frame=in_frame,
        out_frame=out_frame,
        time_step = 10.0, #s
        termination_check = False,
        save_massive_states = True,
    )
    if settings is not None:
        settings.update(reb_settings)
    else:
        settings = reb_settings

    prop = sorts.propagator.Rebound(
        kernel = kernel, 
        settings = settings,
    )

    states, massive_states = prop.propagate(t, state, epoch)

    return states, massive_states, t



states_prop, massive_states, t = propagate_pre_encounter(
    states_ITRS, 
    epoch, 
    in_frame = 'ITRS', 
    out_frame = 'HCRS', 
    kernel = kernel, 
    settings = dict(tqdm=True),
)
# states_prop, massive_states, t = sorts.propagate_pre_encounter(
#     states_ITRS, 
#     epoch, 
#     in_frame = 'ITRS', 
#     out_frame = 'HCRS', 
#     termination_check = sorts.distance_termination(dAU = 0.01), #hill sphere of Earth in AU
#     kernel = kernel, 
#     settings = dict(tqdm=True),
# )

states_end = np.squeeze(states_prop[:,-1,:])

states_HMC = sorts.frames.convert(
    epoch + TimeDelta(t[-1], format='sec'),
    states_end, 
    in_frame = 'HCRS', 
    out_frame = 'HeliocentricMeanEcliptic',
)

orb = pyorb.Orbit(
    M0 = pyorb.M_sol,
    direct_update=True,
    auto_update=True,
    degrees = True,
    num = num,
)
orb.cartesian = states_HMC

orb_t = pyorb.Orbit(
    M0 = pyorb.M_sol,
    direct_update=True,
    auto_update=True,
    degrees = True,
    num = len(t),
)

plt.rc('text', usetex=True)

axis_labels = ["$a$ [AU]","$e$ [1]","$i$ [deg]","$\\omega$ [deg]","$\\Omega$ [deg]", "$\\nu$ [deg]" ]
scale = [1/pyorb.AU] + [1]*5

fig = plt.figure(figsize=(15,15))
for i in range(6):
    ax = fig.add_subplot(231+i)
    ax.hist(orb.kepler[i,:]*scale[i])
    ax.set_ylabel('Frequency')
    ax.set_xlabel(axis_labels[i])
fig.suptitle(f'Elements at pre-encounter {t[-1]/3600} h')


# fig, axes = plt.subplots(2,3,figsize=(15,15))
# axes = axes.flatten()

# for j in range(0,num,10):

#     states_tmp_ = sorts.frames.convert(
#         epoch + TimeDelta(t, format='sec'),
#         np.squeeze(states_prop[:,:,j]), 
#         in_frame = 'HCRS', 
#         out_frame = 'HeliocentricMeanEcliptic',
#     )
#     orb_t.cartesian = states_tmp_

#     for i in range(6):
#         axes[i].plot(t/3600.0, orb_t.kepler[i,:]*scale[i], "-b", alpha=0.1)

# for i in range(6):
#     axes[i].set_xlabel('Time [h]')
#     axes[i].set_ylabel(axis_labels[i])
# fig.suptitle('Propagation to pre-encounter elements')


plt.show()