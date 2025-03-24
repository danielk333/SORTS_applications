import numpy as np

import sorts

prop = sorts.propagator.SGP4(
    settings=dict(
        out_frame="ITRS",
        tle_input=True,
    ),
)
radar = sorts.get_radar("eiscat3d", "stage1-array")

print(prop)

l1 = "1     5U 58002B   20251.29381767 +.00000045 +00000-0 +68424-4 0  9990"
l2 = "2     5 034.2510 336.1746 1845948 000.5952 359.6376 10.84867629214144"

# JD epoch calculated from lines
epoch = 2459099.79381767

t = 1800 + np.arange(10) * 0.2

states_tle = prop.propagate(t, [l1, l2])

rvec = states_tle[:3, :] - radar.tx[0].ecef[:, None]
r = np.linalg.norm(rvec, axis=0)

rn = rvec / r
v = np.sum(states_tle[3:, :] * rn, axis=0)

print(r * 1e-3)
print(v * 1e-3)

a = np.diff(v) / np.diff(t)

print("acceleration m/s^2")
print(a)
