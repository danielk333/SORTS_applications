import subprocess
import pickle

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import TimeDelta
from tqdm import tqdm

import sorts

plt.style.use("dark_background")


try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    class COMM_WORLD:
        rank = 0
        size = 1

        def barrier(self):
            pass
    comm = COMM_WORLD()


def create_movie(path, radar, controller, population, t_frames, epoch, clobber=False):
    plot_folder = path / "frames"
    anim_file = path / "animation.mp4"
    plot_folder.mkdir(exist_ok=True)

    render_frames(path, radar, controller, population, t_frames, epoch, clobber=clobber)
    if not anim_file.is_file() or clobber:
        if comm.rank == 0:
            render_movie(plot_folder, anim_file)


def render_frames(path, radar, controller, population, t_frames, epoch, clobber=False):
    data_folder = path / "data"
    plot_folder = path / "frames"

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")

    states = {}
    passes = {}

    detected = {}

    for obj in population:
        output_prop = data_folder / f"{obj.oid}.npy"
        output_pass = data_folder / f"{obj.oid}.pickle"
        states[obj.oid] = np.load(output_prop)
        with open(output_pass, "rb") as fh:
            passes[obj.oid] = pickle.load(fh)

        allps = passes[obj.oid]
        detected[obj.oid] = None
        for txp in allps:
            for rxp in txp:
                for ps in rxp:
                    zang = ps.zenith_angle(radians=False)[0]
                    best_ind = np.argmin(zang)
                    if (
                        zang[best_ind] < 20
                        and (ps.t[best_ind] > np.min(controller.t) and ps.t[best_ind] < np.max(controller.t))
                    ):
                        detected[obj.oid] = ps.t[best_ind]

    controller.passes = passes
    controller.states = states

    tasks = np.arange(comm.rank, len(t_frames), comm.size)

    comm.barrier()
    pbar = tqdm(desc="Rendering", total=len(tasks), position=comm.rank)

    trail_behind = 4
    trail_alpha = 0.4
    for ti, t, radm in zip(tasks, t_frames[tasks], controller(t_frames[tasks])):
        plot_file = plot_folder / f"t_int{ti}.png"

        if plot_file.is_file() and not clobber:
            pbar.update(1)
            continue

        t0 = ti - trail_behind if ti > trail_behind else 0
        ax.clear()
        for oid, state in states.items():
            c = "c"
            if detected[oid] is not None:
                if detected[oid] < t:
                    c = "r"
            ax.plot(state[0, ti], state[1, ti], state[2, ti], "o" + c)
            ax.plot(
                state[0, t0:(ti + 1)],
                state[1, t0:(ti + 1)],
                state[2, t0:(ti + 1)],
                "-w",
                alpha=trail_alpha,
            )

        teme_txs = []
        for tx in radar.tx:
            teme_tx = sorts.frames.convert(
                epoch + TimeDelta(t, format="sec"),
                np.concatenate([tx.ecef, np.zeros((3,))]),
                in_frame="ITRS",
                out_frame="TEME",
            )
            teme_txs.append(teme_tx)
            ax.plot([teme_tx[0]], [teme_tx[1]], [teme_tx[2]], "og")

        # teme_rxs = []
        # for rx in radar.rx:
        #     teme_rx = sorts.frames.convert(
        #         epoch + TimeDelta(t, format="sec"),
        #         np.concatenate([rx.ecef, np.zeros((3,))]),
        #         in_frame="ITRS",
        #         out_frame="TEME",
        #     )
        #     teme_rxs.append(teme_rx)
        #     ax.plot([teme_rx[0]], [teme_rx[1]], [teme_rx[2]], "xb")

        pointing_ecef, reception_ecef = radm

        if t > np.min(controller.t) and t < np.max(controller.t):
            for txi, tx in enumerate(radar.tx):
                teme_tx = teme_txs[txi]

                point = pointing_ecef[txi] + tx.ecef[:, None]
                teme_point = sorts.frames.convert(
                    epoch + TimeDelta(t, format="sec"),
                    np.concatenate([point, np.zeros_like(point)], axis=0),
                    in_frame="ITRS",
                    out_frame="TEME",
                )

                for pi in range(teme_point.shape[1]):
                    ax.plot(
                        [teme_tx[0], teme_point[0, pi]],
                        [teme_tx[1], teme_point[1, pi]],
                        [teme_tx[2], teme_point[2, pi]],
                        "g-",
                    )
            # for rxi, rx in enumerate(radar.rx):
            #     teme_rx = teme_rxs[rxi]
            #     point = reception_ecef[rxi] + rx.ecef[:, None]
            #     teme_point = sorts.frames.convert(
            #         epoch + TimeDelta(t, format="sec"),
            #         np.concatenate([point, np.zeros_like(point)], axis=0),
            #         in_frame="ITRS",
            #         out_frame="TEME",
            #     )
            #     for pi in range(teme_point.shape[1]):
            #         ax.plot(
            #             [teme_point[0, pi], teme_rx[0]],
            #             [teme_point[1, pi], teme_rx[1]],
            #             [teme_point[2, pi], teme_rx[2]],
            #             "b-",
            #         )

        sorts.plotting.transformed_grid_earth(
            ax, frame="TEME", time=epoch + TimeDelta(t, format="sec"), color="w",
        )
        dx = 7000e3
        ax.set_xlim([-dx, dx])
        ax.set_ylim([-dx, dx])
        ax.set_zlim([-dx, dx])
        ax.view_init(elev=45, azim=-45)
        fig.savefig(plot_file, bbox_inches="tight")
        pbar.update(1)
    pbar.close()
    plt.close(fig)


def render_movie(plot_folder, anim_file):
    try:
        subprocess.check_call(
            f"ffmpeg -start_number 0 -r 25 -i t_int%d.png -vcodec libx264 -crf 22 {str(anim_file.resolve())}",
            cwd=str(plot_folder.resolve()),
            shell=True,
        )
    except subprocess.CalledProcessError as e:
        print(e)
        print(
            "Could not create movie from animation frames... probably ffmpeg is missing"
        )
