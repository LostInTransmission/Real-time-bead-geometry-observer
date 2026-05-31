import numpy as np
from scipy.fft import fft

def calculate_spikes(y, z, threshold=1.2):
    if len(y) < 2 or len(z) < 2:
        return 0
    dy = np.diff(y)
    dz = np.diff(z)
    dy_safe = np.where(dy == 0, 1e-6, dy)
    grad = np.abs(dz / dy_safe)
    return int(np.sum(grad > threshold))

def calculate_dft_noise(z, cutoff=0.25):
    if len(z) < 4:
        return 0.0
    z_centered = z - np.mean(z)
    z_f = np.abs(fft(z_centered))
    n = len(z_f)
    half = n // 2
    idx = max(1, int(half * cutoff))
    low_f = np.sum(z_f[1:idx])
    high_f = np.sum(z_f[idx:half])
    total = low_f + high_f
    if total > 0:
        return (high_f / total) * 100.0
    return 0.0

def calculate_lag1(z):
    if len(z) < 2:
        return 0.0
    z_centered = z - np.mean(z)
    numerator = np.sum(z_centered[:-1] * z_centered[1:])
    denominator = np.sum(z_centered**2)
    if denominator > 0:
        return numerator / denominator
    return 0.0