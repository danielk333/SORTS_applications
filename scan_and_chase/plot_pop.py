import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from astropy.time import Time, TimeDelta

from scheduler import ScanAndChase
import detect

import sorts

pop_cache_path = pathlib.Path('./sample_population.h5')

pop_kw = dict(
    propagator = sorts.propagator.SGP4,
    propagator_options = dict(
        settings = dict(
            in_frame='TEME',
            out_frame='ITRS',
        ),
    ),
)

pop = sorts.Population.load(pop_cache_path, **pop_kw)

logbins = np.linspace(np.log10(pop['d'].min()),np.log10(pop['d'].max()),10)

fig, axes = plt.subplots(1,2)
axes[0].hist(pop['d'], bins=10.0**logbins)
axes[0].set_xscale('log')
axes[0].set_xlabel('Diameter [m]')

axes[1].plot(pop['a']*1e-3, pop['i'], '.')
axes[1].set_xlabel('Semi-major axis [km]')
axes[1].set_ylabel('Inclination [deg]')

plt.show()