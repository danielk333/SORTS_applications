import pathlib
import sorts
import matplotlib.pyplot as plt

path = pathlib.Path("/home/danielk/data/master_2009/celn_20090501_00.sim")

pop = sorts.population.master_catalog(path)


def get_percent(dlim):
    pop_factor = sorts.population.master_catalog_factor(pop, treshhold=dlim, seed=1234)
    pop_factor.filter("a", lambda x: x < 8e6)

    tot0 = len(pop_factor)
    pop_factor.filter("i", lambda x: x > 65.0)

    tot1 = len(pop_factor)
    print(f"{dlim=} m: {tot0=}, {tot1=}, Percentage={tot1/tot0*100:.2f}")


get_percent(0.01)
get_percent(0.07)
get_percent(1.0)

pop_factor = sorts.population.master_catalog_factor(pop, treshhold=0.01, seed=1234)
pop_factor.filter("a", lambda x: x < 8e6)

fig, ax = plt.subplots()
ax.hist(pop_factor["i"], bins=70)
ax.set_xlabel("Inclination [deg]")
ax.set_ylabel("Distribution")
ax.set_title("MASTER 2009 LEO distribution > 1 cm")
ax.set_xlim(0, 120)
plt.show()
