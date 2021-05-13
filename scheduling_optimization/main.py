#!/usr/bin/env python
import pathlib
import numpy as np
import matplotlib.pyplot as plt

import sorts

#local
import od
import config

err_pth = pathlib.Path('/home/danielk/IRF/IRF_GITLAB/SORTS/examples/data/')

print('Scheduling for object:')
print(config.obj)

print('First set of passes [all rx stations]')
for ps in config.passes:
    print(ps)

config.scheduler.set_observations(num=10, frac=1)

Sigma_orb, data = config.scheduler.determine_orbit()

fig, ax = sorts.plotting.observed_parameters(
    [data[0]],
    passes=[config.passes[0]], 
    snrdb_lim = 10.0
)
plt.show()