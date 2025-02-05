import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import geopandas as gpd
from pathlib import Path

import sorts
import pyant.coordinates as coord

radar = sorts.radars.nostra

plt.style.use("dark_background")

gpkd_geo_datas = [
    Path("/home/danielk/data/gpkg/gadm41_NOR.gpkg"),
    Path("/home/danielk/data/gpkg/gadm41_SWE.gpkg"),
    Path("/home/danielk/data/gpkg/gadm41_FIN.gpkg"),
]

G_ROT = coord.rot_mat_y(-11, degrees=True) @ coord.rot_mat_z(-11, degrees=True)
# G_ROT = np.eye(3)

countries_itrs = []
for geo_data in gpkd_geo_datas:
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
passes = sorts.passes.find_passes(t, states, radar.tx, cache_data=False)

np.random.seed(384783)
send_range = 2000e3
scan_ranges = np.linspace(300e3, 2000e3, 100)
plot_range = 700e3

fig = plt.figure(figsize=(3*3, 4*3))
ax = fig.add_subplot(111, projection="3d")

tx_pts = [
    (0, passes[0].inds[len(passes[0].inds)//2]),
    (1, passes[1].inds[len(passes[1].inds)//2]),
    (2, passes[3].inds[len(passes[3].inds)//2]),
]
points = []
for tx_ind, ps_ind in tx_pts:
    ecef_p = states[:3, ps_ind]
    ecef_rel_p = ecef_p - radar.tx[tx_ind].ecef
    points.append(
        (tx_ind, ps_ind, ecef_rel_p/np.linalg.norm(ecef_rel_p))
    )

G_states = G_ROT @ states[:3, :]

for tx_ind, ps_ind, point in points:
    tx = radar.tx[tx_ind]
    txecef = G_ROT @ tx.ecef
    ax.plot(txecef[0], txecef[1], txecef[2], "or")
    send_point = tx.ecef + point * send_range
    send_point = G_ROT @ send_point
    ax.plot(
        [txecef[0], send_point[0]],
        [txecef[1], send_point[1]],
        [txecef[2], send_point[2]],
        "r-",
        lw=2,
    )
    ax.plot(
        [G_states[0, ps_ind]],
        [G_states[1, ps_ind]],
        [G_states[2, ps_ind]],
        "or",
        markersize=10,
    )
    for rx_ind, rx in enumerate(radar.rx):
        if (tx_ind, rx_ind) in radar.joint_stations:
            continue
        rxecef = G_ROT @ rx.ecef
        scan_point = tx.ecef[:, None] + point[:, None] * scan_ranges[None, :]
        scan_point = G_ROT @ scan_point
        for i in range(scan_point.shape[1]):
            ax.plot(
                [rxecef[0], scan_point[0, i]],
                [rxecef[1], scan_point[1, i]],
                [rxecef[2], scan_point[2, i]],
                "g-",
                lw=1,
                alpha=0.3,
            )

for ps in passes:
    ax.plot(
        G_states[0, ps.inds],
        G_states[1, ps.inds],
        G_states[2, ps.inds],
        ls="-", c="c",
    )

for citrs in countries_itrs:
    citrs = G_ROT @ citrs
    ax.plot(
        citrs[0, :],
        citrs[1, :],
        citrs[2, :],
        ls="-", c="w",
    )

# ax.grid(visible=False)
# ax.axis(False)
ax.axis(
    np.array(
        [
            rxecef[0] - plot_range,
            rxecef[0] + plot_range,
            rxecef[1] - plot_range,
            rxecef[1] + plot_range,
            rxecef[2] - plot_range,
            rxecef[2] + plot_range,
        ]
    )
)
# ax.view_init(11, 11)

ax.view_init(0, 0)

# ax.axis((
#     1458796.2210443916, 2858796.2210443914,
#     72668.25422213916, 1472668.2542221393,
#     5515769.194599953, 6915769.194599953,
# ))
ax.set_box_aspect((4, 3, 4))

fig.savefig("plots/nostra_still.jpg")
fig.savefig("plots/nostra_still.png", dpi=300)
plt.savefig("plots/nostra_still.svg", format = 'svg', dpi=300)

plt.show()
