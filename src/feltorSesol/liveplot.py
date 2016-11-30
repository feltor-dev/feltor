#!/usr/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

"""
Plot probe data and radial profiles of current feltor simulation
"""


fig = plt.figure(figsize=(12,12))
ax_probe = fig.add_subplot(121)
ax_prof = fig.add_subplot(122)

num_probes = 7

plt.ion()
plt.show()

while(True):
    ax_prof.cla()
    for n in np.arange(num_probes):
        try:
            df_probe = np.loadtxt('probe_%03d.dat' % n, skiprows=1)
            ax_probe.plot(df_probe[:, 0], df_probe[:, 1]+n*2, 'k')
            ax_probe.plot(df_probe[:, 0], df_probe[:, 3]+n*2, 'r')
        except:
            continue

    try:
        prof_ne = np.loadtxt('ne_prof.dat', skiprows=1)
        prof_phi = np.loadtxt('phi_prof.dat', skiprows=1)

        ax_prof.plot(prof_ne[1:])
        ax_prof.plot(prof_phi[1:])
    except:
        continue

    plt.draw()


# End of file liveplot.py
