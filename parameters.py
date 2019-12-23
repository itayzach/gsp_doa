import numpy as np

# Parameters
N = 41
n = np.arange(0, N)    # samples
M = 6
m = np.arange(1, M+1)  # sensors
f0 = 1e3               # singletone [Hz]
w0 = 2 * np.pi * f0    # [rad / sec]
fs = 2e3               # sampling rate [Hz]
Amp = 1                # amplitude [V]
delta = 0.1            # uniform spacing between sensors [m]
c = 340                # speed of sound [m / s]

theta_d = 70.3  # [degrees]

plots_dir = 'gsp_doa/plots'
