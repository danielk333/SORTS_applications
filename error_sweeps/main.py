#!/usr/bin/env python

'''
Adding random errors
================================

'''
import pathlib

import numpy as np
import matplotlib.pyplot as plt

import sorts.errors as errors
import sorts
eiscat3d = sorts.radars.eiscat3d

pth = pathlib.Path('/home/danielk/IRF/IRF_GITLAB/SORTS/examples/data')

print(f'Caching error calculation data to: {pth}')

#eiscat3d.tx[0].bandwidth = 10000.0
err = errors.LinearizedCodedIonospheric(eiscat3d.tx[0], seed=123, cache_folder=pth)

num2 = 400

ranges = [300e3, 1000e3, 2000e3]

range_rates = np.linspace(0, 10e3, num=num2)
snrs_db = np.linspace(14,50,num=num2)
snrs = 10.0**(snrs_db*0.1)

v_std = err.range_rate_std(snrs)

fig, axes = plt.subplots(1,2)

for j in range(len(ranges)):
    range_ = np.zeros_like(snrs)
    range_[:] = ranges[j]
    r_std = err.range_std(range_, snrs)
    axes[0].semilogy(snrs_db, r_std, label=f'Range: {ranges[j]*1e-3:.1f} km')

axes[0].set_xlabel('SNR [dB]')
axes[0].set_ylabel('Range standard deviation [m]')


axes[0].legend()
axes[1].plot(snrs_db, v_std)
axes[1].set_xlabel('SNR [dB]')
axes[1].set_ylabel('Range-rate standard deviation [m/s]')

plt.tight_layout()
plt.show()