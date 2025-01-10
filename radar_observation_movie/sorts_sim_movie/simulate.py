#!/usr/bin/env python
# import pathlib
import pickle

import numpy as np
from tqdm import tqdm
from astropy.time import TimeDelta

import sorts


def run_simulation(path, radar, population, t_frames, epoch, clobber=False):
    data_folder = path / "data"
    data_folder.mkdir(exist_ok=True)
    pbar = tqdm(desc="Propagating", total=len(population))
    for obj in population:
        output_prop = data_folder / f"{obj.oid}.npy"
        output_pass = data_folder / f"{obj.oid}.pickle"

        if output_prop.is_file() and not clobber:
            continue
        states = obj.get_state(t_frames)
        itrs_states = sorts.frames.convert(
            epoch + TimeDelta(t_frames, format="sec"),
            states,
            in_frame="TEME",
            out_frame="ITRS",
        )
        passes = radar.find_passes(t_frames, itrs_states, cache_data = True)
        np.save(output_prop, states)
        with open(output_pass, "wb") as fh:
            pickle.dump(passes, fh)
        pbar.update(1)
    pbar.close()
