import os
import argparse
import importlib
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--filter', type=str, choices=['SOR', 'Median', 'DBSCAN'], default='SOR')
    args = parser.parse_args()

    if args.filter == 'SOR':
        module_name = 'online_pipeline'
    else:
        module_name = f'online_pipeline_{args.filter}'

    pipeline_module = importlib.import_module(module_name)
    BeadGeometryObserver = pipeline_module.BeadGeometryObserver

    params = {
        'PRINT_SPEED': 10.0,
        'TRIM_START': 10.0,
        'TRIM_END': 10.0,
        'X_MIN': -2.0,
        'X_MAX': 8.0,
        'PERC_MIN': 5,
        'ULS': 0.9,
        'MIN_PTS': 15,
        'MIN_WIDTH': 0.2,
        'MIN_CURV': 0.2,
        'APEX_TOL': 0.5,
        'SOR_K': 7,
        'SOR_MULT': 0.3,
        'MEDIAN_WINDOW': 11,
        'MEDIAN_THRESHOLD': 0.16,
        'CONN_DX': 0.4,
        'CONN_DY': 2.5,
        'CONN_DZ': 0.3,
        'EPS': 1,
        'MIN_SAMPLES': 20,
        'MIN_CLUSTER_SIZE': 55
    }

    observer = BeadGeometryObserver(params)
    df = pd.read_csv(args.file)

    os.makedirs("Results", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.file))[0]
    apexes_file = os.path.join("Results", f"{base_name}_Apexes.csv")
    heights_file = os.path.join("Results", f"{base_name}_CalculatedHeights.csv")

    apexes_data = []
    heights_data = []

    for layer_index, layer_df in df.groupby('Layer'):
        layer_index = int(layer_index)
        scan_profiles = []
        
        for scan_id, scan_df in layer_df.groupby('ScanID'):
            px = scan_df['X'].values.astype(np.float32)
            pz = scan_df['Z'].values.astype(np.float32)
            ts = float(scan_df['Timestamp'].iloc[0])
            scan_profiles.append((px, pz, ts))
            
        current_z_shift = 0.0
        if layer_index > 1:
            current_z_shift = (layer_index - 1) * params['ULS']
        
        result = observer.process_layer(layer_index, scan_profiles, current_z_shift)
        
        if isinstance(result, tuple):
            height, apexes = result
        else:
            height = result
            apexes = []

        heights_data.append({'Layer': layer_index, 'CalculatedHeight': height})
        
        for ap in apexes:
            ap_copy = ap.copy()
            ap_copy['Layer'] = layer_index
            apexes_data.append(ap_copy)

    pd.DataFrame(heights_data).to_csv(heights_file, index=False)
    
    if apexes_data:
        cols = ['Layer', 'X', 'Y', 'Z']
        available_cols = [c for c in cols if c in apexes_data[0]] + [c for c in apexes_data[0] if c not in cols]
        pd.DataFrame(apexes_data)[available_cols].to_csv(apexes_file, index=False)
    else:
        pd.DataFrame(columns=['Layer', 'X', 'Y', 'Z']).to_csv(apexes_file, index=False)

if __name__ == "__main__":
    main()