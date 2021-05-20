# Scan and chase simulation

## Install

First install `sorts`.

```bash
git clone https://github.com/danielk333/SORTS
cd sorts
pip install .
cd ..
```

Then get this repository

```bash
git clone https://github.com/danielk333/SORTS_applications
cd SORTS_applications
cd scan_and_chase
```

## Running the simulation

To run the simulation use the following command-line syntax

```bash
python simulation.py [sparse/full] [/run/norun]
```

Here `sparse` refer to a sparse chase strategy and `full` to using all radar time for the chase.
If `[/run/norun]` is left empty, default is `run`. If `run` is given, all objects are simulated and saved to cache.
If `norun` is given, no operation is performed and the simulation object is left in the context.

The `norun` option is used for interacting with the simulation in a live context, like ipython.

For example, in ipython one could simulate an individual object (index 4) by
```python
%run simulation.py sparse norun
sim.get_data(4)
```