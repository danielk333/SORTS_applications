import scipy.constants
import numpy as np
import matplotlib.pyplot as plt

import sorts

# gestra 1kw / antenna
# ant 10 cm area
# size = sqrt(num)

lam = scipy.constants.c / 3.4e9
print(lam)

alpha = 4
ant_d = alpha*lam*0.5
ant_pw = 0.5e3
size_vec = np.arange(100, 2000)
eff = 0.8


def calc_params(num):
    direc = 16 * num * ant_d**2 / lam**2
    gain = direc * eff

    return ant_pw * num, gain


def get_snr_vec(r, num, pulses, diameter):

    pw, gain = calc_params(size_vec)
    snr = sorts.signals.hard_target_snr(
        gain_tx=gain,
        gain_rx=gain,
        wavelength=lam,
        power_tx=pw,
        range_tx_m=r,
        range_rx_m=r,
        diameter=diameter,
        bandwidth=1.0 / (1e-3 * pulses),
        rx_noise_temp=150.0,
        radar_albedo=1.0,
    )
    return snr


fig, axes = plt.subplots(2, 2)
axes = axes.flatten()
pw, g = calc_params(size_vec)
diam = 2*ant_d*np.sqrt(size_vec)/np.sqrt(np.pi)

axes[0].plot(pw*1e-3, np.log10(g)*10)
axes[0].set_xlabel("power kw")
axes[0].set_ylabel("gain db")

axes[1].plot(size_vec, np.log10(g)*10)
axes[1].set_xlabel("antenna number")
axes[1].set_ylabel("gain db")

axes[2].plot(size_vec, pw*1e-3)
axes[2].set_xlabel("antenna number")
axes[2].set_ylabel("power kw")

axes[3].plot(size_vec, diam)
axes[3].set_xlabel("antenna number")
axes[3].set_ylabel("array diameter")

fig, ax = plt.subplots()
for diam in np.arange(1, 10)*1e-2:
    snr = get_snr_vec(r=1000e3, num=size_vec, pulses=20, diameter=diam)
    snrdb = np.log10(snr)*10

    ax.plot(size_vec, snrdb, label=f"{diam*1e2:.1f} cm")
ax.axhline(0, c="r")
ax.legend()
ax.set_xlabel("antenna number")
ax.set_ylabel("snr dB")
ax.set_title("r=1000e3, pulses=20")

snr = get_snr_vec(r=24000e3, num=size_vec, pulses=40, diameter=100e-2)
snrdb = np.log10(snr)*10

fig, ax = plt.subplots()
ax.plot(size_vec, snrdb)
ax.axhline(0, c="r")
ax.set_xlabel("antenna number")
ax.set_ylabel("snr dB")
ax.set_title("r=24000e3, pulses=40")

plt.show()
