import numpy as np
from matplotlib import pyplot as plt


def linear_frequency_sweep(t, f1, f2, T, A=1.0):
    """
    Generates a linear frequency sweep excitation (chirp).

    Parameters:
        t : float or array_like
            Time or array of time points (0 <= t <= T)
        f1 : float
            Initial frequency in Hz
        f2 : float
            Final frequency in Hz
        T : float
            Duration of sweep in seconds
        A : float
            Amplitude of excitation

    Returns:
        float or ndarray: Excitation force at time t
    """
    t = np.asarray(t)
    phi = 2 * np.pi * (f1 * t + ((f2 - f1) / (2 * T)) * t**2)
    # Instantaneous frequency
    f_inst = f1 + (f2 - f1) * t / T
    dphi_dt = 2 * np.pi * f_inst

    # Displacement and velocity
    x = A * np.sin(phi)
    v = A * np.cos(phi) * dphi_dt
    return x, v



