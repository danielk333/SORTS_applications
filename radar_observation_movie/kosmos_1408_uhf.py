"""

example run

python kosmos_1408_uhf.py simulate /home/danielk/data/sorts/kosmos_1408_movie

"""

import pathlib
import argparse
import numpy as np
from astropy.time import Time, TimeDelta

import sorts

import sorts_sim_movie

np.random.seed(289784)

parser = argparse.ArgumentParser(description="Simulate and create a movie of the simulation")
parser.add_argument(
    "action",
    choices=["simulate", "render"],
    help="which step to perform",
)
parser.add_argument(
    "path",
    type=pathlib.Path,
    help="target output folder",
)
parser.add_argument("-c", "--clobber", action="store_true")

args = parser.parse_args()
clobber = args.clobber


OUTPUT = args.path
OUTPUT.mkdir(exist_ok=True)
tle_file = OUTPUT / "kosmos-1408-2021-11-15.tle"


radar = sorts.radars.eiscat_uhf
radar.rx = radar.rx[:1]

scan = sorts.radar.scans.Beampark(
    azimuth=90.0,
    elevation=75.0,
    dwell=0.1,
)

ecef_pointing = sorts.frames.azel_to_ecef(
    radar.tx[0].lat,
    radar.tx[0].lon,
    radar.tx[0].alt,
    az=90.0,
    el=75.0,
    degrees=True,
)
ecef_pointing.shape = (3, 1)

ecef_pointing *= 1000e3

pop = sorts.population.tle_catalog(tle_file, kepler=True)
pop.unique()

# print(f"TLEs = {len(pop)}")

tle_obj = pop.get_object(0)
# print(tle_obj)

sim_epoch = Time("2021-11-15 02:47:31.5", format="iso", scale="utc")

# obs_start = Time("2021-11-23 10:00", format="iso", scale="utc")
# obs_end = obs_start + TimeDelta(3600.0 * 7, format="sec")
# t_obs_step = 10.0
# t_prop_step = 10*60.0

obs_start = sim_epoch + TimeDelta(3600.0 * 6, format="sec")
obs_end = obs_start + TimeDelta(3600.0 * 7, format="sec")
t_obs_step = 20.0
t_prop_step = 60.0

obs_start_sec = (obs_start - sim_epoch).sec
obs_end_sec = (obs_end - sim_epoch).sec

# fragments = 1000
fragments = 100
log10d_mu = -1.1
log10d_std = 0.2
log10m_mu = -0.9
log10m_std = 0.3
delta_v_std = 200.0  # m/s

t_prop0 = np.arange(0, 3600, t_obs_step)
t_prop = np.arange(3600, obs_start_sec, t_prop_step)
t_obs = np.arange(obs_start_sec, obs_end_sec, t_obs_step)
t_frames = np.concatenate([
    t_prop0,
    t_prop,
    t_obs,
])
# print(f"t obs frames: {len(t_obs)}")
# print(f"t prop frames: {len(t_prop)}")


class Controller:
    def __init__(self, t):
        self.t = t

    def __call__(self, times):
        for t in times:
            yield [ecef_pointing, ], [ecef_pointing, ]


controller = Controller(t_obs)


tle_obj.out_frame = "TEME"
state0 = tle_obj.get_state(sim_epoch).flatten()
mjd0 = tle_obj.epoch.mjd

# Random sample size distribution
d = 10.0 ** (np.random.randn(fragments) * log10d_std + log10d_mu)
A = np.pi * (d * 0.5) ** 2

m = 10.0 ** (np.random.randn(fragments) * log10m_std + log10m_mu)

A_m_ratio = A / m


states0 = np.empty((6, fragments), dtype=np.float64)
states0[:3, :] = state0[:3, None]
states0[3:, :] = state0[3:, None] + np.random.randn(3, fragments) * delta_v_std

cart_names = ["x", "y", "z", "vx", "vy", "vz"]
prop_names = ["A", "m", "C_D", "C_R"]
fields = ["oid"] + cart_names + ["mjd0"] + prop_names
dtypes = ["int"] + ["float64"] * len(cart_names) + ["float64"] + ["float64"] * len(prop_names)


# orekit_data = OUTPUT / "orekit-data-master.zip"
# if not orekit_data.is_file():
#     sorts.propagator.Orekit.download_quickstart_data(orekit_data, verbose=True)
# settings = dict(
#     in_frame="TEME",
#     out_frame="TEME",
#     solar_activity_strength="WEAK",
#     integrator="DormandPrince853",
#     min_step=0.001,
#     max_step=240.0,
#     position_tolerance=20.0,
#     earth_gravity="HolmesFeatherstone",
#     gravity_order=(10, 10),
#     solarsystem_perturbers=["Moon", "Sun"],
#     drag_force=True,
#     atmosphere="DTM2000",
#     radiation_pressure=True,
#     solar_activity="Marshall",
#     constants_source="WGS84",
# )
# propagator = sorts.propagator.Orekit
# propagator_options = dict(
#     orekit_data=orekit_data,
#     settings=settings,
# )


settings = dict(
    in_frame="TEME",
    out_frame="TEME",
)
propagator = sorts.propagator.SGP4
propagator_options = dict(
    settings=settings,
)

population = sorts.Population(
    fields=fields,
    dtypes=dtypes,
    space_object_fields=["A", "m", "C_D", "C_R"],
    state_fields=["x", "y", "z", "vx", "vy", "vz"],
    epoch_field={"field": "mjd0", "format": "mjd", "scale": "utc"},
    propagator=propagator,
    propagator_options=propagator_options,
    propagator_args={},
)

population.allocate(fragments)
population.data["oid"] = np.arange(fragments)
for ind, key in enumerate(cart_names):
    population.data[key] = states0[ind, :]
population.data["mjd0"] = mjd0
population.data["A"] = A
population.data["m"] = m
population.data["C_D"] = 1
population.data["C_R"] = 1


if args.action == "simulate":
    sorts_sim_movie.run_simulation(OUTPUT, radar, population, t_frames, sim_epoch, clobber=clobber)
elif args.action == "render":
    sorts_sim_movie.create_movie(OUTPUT, radar, controller, population, t_frames, sim_epoch, clobber=clobber)
