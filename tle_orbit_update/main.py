import pathlib
import argparse

import h5py
import numpy as np
import scipy.optimize

from astropy.time import Time, TimeDelta
import astropy.coordinates as coord
import astropy.units as units

import sgp4
from sgp4.api import Satrec
import sgp4.earth_gravity
from sgp4 import exporter

import matplotlib.pyplot as plt

np.random.seed(32847)

GRAV_IND = sgp4.api.WGS84
GRAV_MODEL = sgp4.earth_gravity.wgs84
RHO0 = 2.461e-5 / 6378.135e3  # kg/m^2/m
SGP4_MJD0 = Time("1949-12-31 00:00:00", format="iso", scale="ut1").mjd


def load_tle(path):
    with open(path, "r") as fh:
        lines = fh.readlines()
    lines = [x for x in lines if len(lines) > 0]
    assert len(lines) == 2
    return lines


def load_observations(path):
    with h5py.File(path, "r") as hf:
        times = Time(hf["t"][()], format="unix", scale="utc")
        r = hf["r"][()]
        v = hf["v"][()]
        rx_ecef = hf["rx_ecef"][()]
        tx_ecef = hf["tx_ecef"][()]
    return times, r, v, rx_ecef, tx_ecef


def main():

    parser = argparse.ArgumentParser(description="Update TLE using observations")
    parser.add_argument(
        "tle", type=pathlib.Path, help="File with TLE",
    )
    parser.add_argument(
        "observations", type=pathlib.Path, help="hdf5 file with observations",
    )
    parser.add_argument(
        "-o", "--output", type=pathlib.Path, default=None, help="Optional output path for new TLE",
    )
    parser.add_argument(
        "-p", "--plot", action="store_true", help="Plot update difference",
    )
    args = parser.parse_args()
    line1, line2 = load_tle(args.tle)
    times, r, v, rx_ecef, tx_ecef = load_observations(args.observations)

    new_tle = update_tle(line1, line2, times, r, v, rx_ecef, tx_ecef, plot=args.plot)

    if args.output is None:
        print("\n".join(new_tle))
    else:
        with open(args.output, "w") as fh:
            fh.write(new_tle[0] + "\n")
            fh.write(new_tle[1])


def line_decode(line):
    if isinstance(line, np.bytes_):
        line = line.astype("U")
    elif not isinstance(line, str):
        try:
            line = line.decode()
        except (UnicodeDecodeError, AttributeError):
            pass

    return line


def generate_measurements(state_ecef, rx_ecef, tx_ecef):
    r_tx = tx_ecef[:, None] - state_ecef[:3, :]
    r_rx = rx_ecef[:, None] - state_ecef[:3, :]

    r_tx_n = np.linalg.norm(r_tx, axis=0)
    r_rx_n = np.linalg.norm(r_rx, axis=0)
    r_sim = r_tx_n + r_rx_n

    v_tx = -np.sum(r_tx * state_ecef[3:, :], axis=0) / r_tx_n
    v_rx = -np.sum(r_rx * state_ecef[3:, :], axis=0) / r_rx_n

    v_sim = v_tx + v_rx

    return r_sim, v_sim


def get_sgp4_satellite(mean_elements, bstar, satnum, epoch, ndot, nddot):
    epoch0 = epoch.mjd - SGP4_MJD0
    satellite = Satrec()
    satellite.sgp4init(
        GRAV_IND,  # gravity model
        "i",  # 'a' = old AFSPC mode, 'i' = improved mode
        satnum,  # satnum: Satellite number
        epoch0,  # epoch: days since 1949 December 31 00:00 UT
        bstar,  # bstar: drag coefficient (/earth radii)
        ndot,  # [IGNORED BY SGP4] ndot: ballistic coefficient (revs/day)
        nddot,  # [IGNORED BY SGP4] nddot: second derivative of mean motion (revs/day^3)
        mean_elements[1],  # ecco: eccentricity
        mean_elements[4],  # argpo: argument of perigee (radians)
        mean_elements[2],  # inclo: inclination (radians)
        mean_elements[5],  # mo: mean anomaly (radians)
        mean_elements[0],  # no_kozai: mean motion (radians/minute)
        mean_elements[3],  # nodeo: right ascension of ascending node (radians)
    )
    return satellite


def get_mean_elements(satellite):
    """Extract the mean elements in SI units
    (a [m], e [1], inc [rad], raan [rad], aop [rad], mu [rad]),
    B-parameter (not bstar) and epoch from a two line element pair.

    NOTE: code taken from SORTS package

    """

    epoch = Time(satellite.jdsatepoch + satellite.jdsatepochF, format="jd", scale="utc")

    mean_elements = np.zeros((6,), dtype=np.float64)

    mean_elements[0] = satellite.no_kozai
    mean_elements[1] = satellite.ecco
    mean_elements[2] = satellite.inclo
    mean_elements[3] = satellite.nodeo
    mean_elements[4] = satellite.argpo
    mean_elements[5] = satellite.mo

    return mean_elements, satellite.bstar, satellite.satnum, epoch, satellite.ndot, satellite.nddot


def sim_measurnment(
    mean_elements, bstar, satnum, epoch, ndot, nddot, times, rx_ecef, tx_ecef
):
    sat = get_sgp4_satellite(mean_elements, bstar, satnum, epoch, ndot, nddot)
    errors, pos, vel = sat.sgp4_array(times.jd1, times.jd2)
    states = np.empty((6, times.size), dtype=np.float64)
    states[:3, ...] = pos.T
    states[3:, ...] = vel.T

    state_p = coord.CartesianRepresentation(states[:3, ...] * units.m)
    state_v = coord.CartesianDifferential(states[3:, ...] * units.m / units.s)
    astropy_states = coord.TEME(state_p.with_differentials(state_v), obstime=times)

    out_states = astropy_states.transform_to(coord.ITRS(obstime=times))

    states[:3, ...] = out_states.cartesian.xyz.to(units.m).value
    states[3:, ...] = out_states.velocity.d_xyz.to(units.m / units.s).value

    r_sim, v_sim = generate_measurements(states, rx_ecef, tx_ecef)
    return r_sim, v_sim


def least_squares(
    mean_elements, bstar, satnum, epoch, ndot, nddot, times, r, v, rx_ecef, tx_ecef, v_weight=1e3
):
    r_sim, v_sim = sim_measurnment(
        mean_elements, bstar, satnum, epoch, ndot, nddot, times, rx_ecef, tx_ecef
    )
    d = np.sum((r_sim - r)**2 + v_weight*(v_sim - v)**2)

    return d


def update_tle(line1, line2, times, r, v, rx_ecef, tx_ecef, plot=False):
    line1, line2 = line_decode(line1), line_decode(line2)
    satellite = Satrec.twoline2rv(line1, line2, GRAV_IND)

    mean_elements0, bstar, satnum, epoch, ndot, nddot = get_mean_elements(satellite)

    bounds = [
        (0, np.inf),
        (0, 1),
        (0, np.pi),
        (0, 2 * np.pi),
        (0, 2 * np.pi),
        (0, 2 * np.pi),
    ]
    result = scipy.optimize.minimize(
        least_squares,
        mean_elements0,
        method="Nelder-Mead",
        args=(bstar, satnum, epoch, ndot, nddot, times, r, v, rx_ecef, tx_ecef),
        bounds=bounds,
        options={
            # "fatol": 1e-7,
        },
    )
    mean_elements = result.x

    new_satellite = get_sgp4_satellite(mean_elements, bstar, satnum, epoch, ndot, nddot)
    new_line1, new_line2 = exporter.export_tle(new_satellite)

    if plot:
        r_sim0, v_sim0 = sim_measurnment(
            mean_elements0, bstar, satnum, epoch, ndot, nddot, times, rx_ecef, tx_ecef
        )
        r_sim, v_sim = sim_measurnment(
            mean_elements, bstar, satnum, epoch, ndot, nddot, times, rx_ecef, tx_ecef
        )
        dt = (times - epoch).sec

        fig, axes = plt.subplots(2, 1)
        axes[0].plot(dt, r_sim0 * 1e-3, label="Initial TLE")
        axes[0].plot(dt, r_sim * 1e-3, label="Updated TLE")
        axes[0].plot(dt, r * 1e-3, ".", label="Measurements")
        axes[0].legend()
        axes[0].set_xlabel("Time + epoch [s]")
        axes[0].set_xlabel("Range [km]")

        axes[1].plot(dt, v_sim0 * 1e-3, label="Initial TLE")
        axes[1].plot(dt, v_sim * 1e-3, label="Updated TLE")
        axes[1].plot(dt, v * 1e-3, ".", label="Measurements")
        axes[1].set_xlabel("Time + epoch [s]")
        axes[1].set_xlabel("Range-rate [km/s]")

        plt.show()

    return new_line1, new_line2


def generate_test_files():
    """To test

    run this to generate test files

    python -c "import main; main.generate_test_files()"

    then compare input and output, diff should be defined by implemented change

    cat tle.txt
    python main.py obs.h5

    """
    line1 = "1 13552U 82092A   21319.03826954  .00002024  00000-0  69413-4 0  9995"
    line2 = "2 13552  82.5637 123.6906 0018570 108.1104 252.2161 15.29390138142807"

    line1, line2 = line_decode(line1), line_decode(line2)
    satellite = Satrec.twoline2rv(line1, line2, GRAV_IND)

    mean_elements, bstar, satnum, epoch, ndot, nddot = get_mean_elements(satellite)
    mean_elements[2] += np.radians(12)  # inc
    mean_elements[5] += np.radians(0.1)  # inc

    satellite = get_sgp4_satellite(mean_elements, bstar, satnum, epoch, ndot, nddot)

    times = epoch + TimeDelta(np.arange(0, 2, 0.2) + 12*3600, format="sec")

    errors, pos, vel = satellite.sgp4_array(times.jd1, times.jd2)
    states = np.empty((6, times.size), dtype=np.float64)
    states[:3, ...] = pos.T
    states[3:, ...] = vel.T

    state_p = coord.CartesianRepresentation(states[:3, ...] * units.m)
    state_v = coord.CartesianDifferential(states[3:, ...] * units.m / units.s)
    astropy_states = coord.TEME(state_p.with_differentials(state_v), obstime=times)

    out_states = astropy_states.transform_to(coord.ITRS(obstime=times))

    states[:3, ...] = out_states.cartesian.xyz.to(units.m).value
    states[3:, ...] = out_states.velocity.d_xyz.to(units.m / units.s).value

    lat = 69.0 + (35 + 11 / 60.0) / 60.0
    lon = 19.0 + (13 + 38 / 60.0) / 60.0
    alt = 86.0

    cord = coord.EarthLocation.from_geodetic(
        lon=lon * units.deg,
        lat=lat * units.deg,
        height=alt * units.m,
    )
    x, y, z = cord.to_geocentric()

    st_ecef = np.empty((3,), dtype=np.float64)
    st_ecef[0] = x.to(units.m).value
    st_ecef[1] = y.to(units.m).value
    st_ecef[2] = z.to(units.m).value

    r_sim, v_sim = generate_measurements(states, st_ecef, st_ecef)

    r_sim += np.random.randn(*r_sim.shape)*150.0
    v_sim += np.random.randn(*v_sim.shape)*15.0

    with h5py.File("obs.h5", "w") as hf:
        hf["t"] = times.unix
        hf["r"] = r_sim
        hf["v"] = v_sim
        hf["rx_ecef"] = st_ecef
        hf["tx_ecef"] = st_ecef
    with open("tle.txt", "w") as fh:
        fh.write(line1 + "\n")
        fh.write(line2)


if __name__ == "__main__":
    main()
