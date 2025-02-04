import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time, TimeDelta
import geopandas as gpd
from pathlib import Path

import sorts

radar = sorts.radars.eiscat3d
plt.style.use("dark_background")

geo_datas = [
    Path("/home/danielk/data/gpkg/gadm41_NOR.gpkg"),
    Path("/home/danielk/data/gpkg/gadm41_SWE.gpkg"),
    Path("/home/danielk/data/gpkg/gadm41_FIN.gpkg"),
]

countries_itrs = []
for geo_data in geo_datas:
    print("loading ", geo_data)
    data_gdf = gpd.read_file(geo_data, layer="ADM_ADM_0")
    for poly in data_gdf["geometry"][0].geoms:
        xx, yy = poly.exterior.coords.xy
        c_itrs = sorts.frames.geodetic_to_ITRS(yy, xx, np.zeros_like(xx), degrees=True)
        countries_itrs.append(c_itrs)


poptions = dict(
    settings=dict(
        in_frame="GCRS",
        out_frame="ITRS",
    ),
)
epoch = Time(53005.0, format="mjd")

scan = sorts.scans.Fence(azimuth=90, num=40, dwell=0.1, min_elevation=30)

obj = sorts.SpaceObject(
    sorts.propagator.SGP4,
    propagator_options=poptions,
    a=7200e3,
    e=0.02,
    i=75,
    raan=86,
    aop=0,
    mu0=60,
    epoch=epoch,
    parameters=dict(
        d=0.1,
    ),
)

t = np.arange(0, 24 * 3600, 30.0)

states = obj.get_state(t)


scan = sorts.scans.RandomUniform()


np.random.seed(384783)
scan_range = 1000e3
plot_range = 300e3

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection="3d")

point = scan.enu_pointing(np.linspace(0, scan.cycle(), num=100))

for rx in radar.rx:
    ax.plot(rx.ecef[0], rx.ecef[1], rx.ecef[2], "or")

scan_point = rx.ecef[:, None] + point * scan_range
for i in range(point.shape[1]):
    ax.plot(
        [rx.ecef[0], scan_point[0, i]],
        [rx.ecef[1], scan_point[1, i]],
        [rx.ecef[2], scan_point[2, i]],
        "g-",
    )

ax.plot(states[0, :], states[1, :], states[2, :], ls="-", c="w")

for citrs in countries_itrs:
    ax.plot(citrs[0, :], citrs[1, :], citrs[2, :], ls="-", c="w")

ax.grid(visible=False)
ax.axis(False)
ax.axis(
    np.array(
        [
            rx.ecef[0] - plot_range,
            rx.ecef[0] + plot_range,
            rx.ecef[1] - plot_range,
            rx.ecef[1] + plot_range,
            rx.ecef[2] - plot_range,
            rx.ecef[2] + plot_range,
        ]
    )
)
ax.view_init(6, 25)

plt.show()
