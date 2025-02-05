import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import geopandas as gpd
from pathlib import Path

import sorts

np.random.seed(384783)

radar = sorts.get_radar("nostra", "example1")

movie_folder = Path("plots/frames")
movie_folder.mkdir(exist_ok=True)

plt.style.use("dark_background")

gpkd_geo_datas = [
    Path("/home/danielk/data/gpkg/gadm41_NOR.gpkg"),
    Path("/home/danielk/data/gpkg/gadm41_SWE.gpkg"),
    Path("/home/danielk/data/gpkg/gadm41_FIN.gpkg"),
]

countries_itrs = []
# for geo_data in gpkd_geo_datas:
#     print("loading ", geo_data)
#     data_gdf = gpd.read_file(geo_data, layer="ADM_ADM_0")

#     for poly in data_gdf["geometry"][0].geoms:
#         xx, yy = poly.exterior.coords.xy
#         c_itrs = sorts.frames.geodetic_to_ITRS(yy, xx, np.zeros_like(xx), degrees=True)
#         countries_itrs.append(c_itrs)

poptions = dict(
    settings=dict(
        in_frame="GCRS",
        out_frame="ITRS",
    ),
)
epoch = Time(53005.0, format="mjd")

scan = sorts.scans.Fence(azimuth=90, num=40, dwell=0.1, min_elevation=30)

master_path = Path("/home/danielk/data/master_2009/celn_20090501_00.sim")
pop = sorts.population.master_catalog(master_path)
pop_factor = sorts.population.master_catalog_factor(pop, treshhold=0.1, seed=1234)
pop_factor.filter("a", lambda x: x < 8e6)

print(f"{len(pop_factor)=}")
exit()
frame_dt = 2.0
fps = 30
seconds_per_second = frame_dt*fps
t = np.arange(0, 1 * 3600, frame_dt)

objects = {}
for obj in pop_factor:
    states = obj.get_state(t)
    passes = sorts.passes.find_passes(t, states, radar.tx, cache_data=False)
    objects[obj.oid] = (states, passes)

send_range = 2000e3
scan_ranges = np.linspace(300e3, 2000e3, 100)
plot_range = 700e3

fig = plt.figure(figsize=(15, 15))
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

for tx_ind, ps_ind, point in points:
    tx = radar.tx[tx_ind]
    ax.plot(tx.ecef[0], tx.ecef[1], tx.ecef[2], "or")
    send_point = tx.ecef + point * send_range
    ax.plot(
        [tx.ecef[0], send_point[0]],
        [tx.ecef[1], send_point[1]],
        [tx.ecef[2], send_point[2]],
        "r-",
        lw=2,
    )
    ax.plot(
        [states[0, ps_ind]],
        [states[1, ps_ind]],
        [states[2, ps_ind]],
        "or",
        markersize=10,
    )
    for rx_ind, rx in enumerate(radar.rx):
        if (tx_ind, rx_ind) in radar.joint_stations:
            continue
        scan_point = tx.ecef[:, None] + point[:, None] * scan_ranges[None, :]
        for i in range(scan_point.shape[1]):
            ax.plot(
                [rx.ecef[0], scan_point[0, i]],
                [rx.ecef[1], scan_point[1, i]],
                [rx.ecef[2], scan_point[2, i]],
                "g-",
                lw=1,
                alpha=0.3,
            )

for ps in passes:
    ax.plot(states[0, ps.inds], states[1, ps.inds], states[2, ps.inds], ls="-", c="c")

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
# ax.view_init(6, 25)
ax.view_init(11, 11)

fig.savefig("plots/nostra_still.jpg")
fig.savefig("plots/nostra_still.png", dpi=300)
plt.savefig("plots/nostra_still.svg", format = 'svg', dpi=300)
plt.show()
