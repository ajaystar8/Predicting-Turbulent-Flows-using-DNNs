import numpy as np

Lx = 4 * np.pi
Ly = 2
Lz = 2 * np.pi

nx = ny = nz = 21

Re = 400  # Reynold's Number
a = 2 * np.pi / Lx  # alpha
b = np.pi / 2.  # beta
g = 2 * np.pi / Lz  # gamma
params = [a, b, g, Re]

x = np.linspace(0, Lx, nx)
y = np.linspace(-1, 1, ny)
z = np.linspace(0, Lz, nz)
time = np.linspace(0, 4000, 4001)  # time array
seqLen = 10

LOOK_BACK = 10
NUM_EPOCHS = 10

HIDDEN_UNITS = 90
