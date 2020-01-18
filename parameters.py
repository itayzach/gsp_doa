import numpy as np

# Parameters
N = 41
n = np.arange(0, N)    # samples
M = 6
m = np.arange(1, M+1)  # sensors
f0 = 1e3               # singletone [Hz]
w0 = 2 * np.pi * f0    # [rad / sec]
fs = 16e3              # sampling rate [Hz]
Amp = 1                # amplitude [V]
c = 340                # speed of sound [m / s]
delta_over_lambda = 0.7
delta = delta_over_lambda*c/f0       # uniform spacing between sensors [m]

theta_d = 70.3  # [degrees]
# In order to ensure signal is labeled False if theta is more than theta_threshold degrees farther than theta_d:
theta_threshold = 0.25  # [degrees]
theta_threshold_for_plot = 1.5

plots_dir = 'gsp_doa/plots'
