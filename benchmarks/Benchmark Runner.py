import os
import time
import pandas as pd
import numpy as np
from calculate_metrics import calculate_spikes, calculate_dft_noise, calculate_lag1

def run_benchmark(filepath):
    if not os.path.exists(filepath):
        print(f"File is not found: {filepath}")
        return

    df = pd.read_csv(filepath)
    
    if 'Y' not in df.columns or 'Z' not in df.columns:
        print("Error: CSV need to include 'Y' & 'Z' with calculated apexes.")
        return

    df_sorted = df.sort_values('Y')
    y = df_sorted['Y'].values.astype(np.float64)
    z = df_sorted['Z'].values.astype(np.float64)

    start_time = time.perf_counter()
    
    spikes = calculate_spikes(y, z, threshold=1.2)
    dft_noise = calculate_dft_noise(z, cutoff=0.25)
    lag1 = calculate_lag1(z)
    
    end_time = time.perf_counter()
    calc_time_ms = (end_time - start_time) * 1000.0

    print(f"--- Benchmarking Results for {filepath} ---")
    print(f"Spikes count:     {spikes}")
    print(f"DFT noise (%):    {dft_noise:.2f}")
    print(f"Lag-1 (r1):       {lag1:.3f}")
    print(f"Время расчета:    {calc_time_ms:.4f} мс")
    print("-" * 50)

if __name__ == "__main__":
    file_path = "../data/single_layer_representative.csv"
    run_benchmark(file_path)