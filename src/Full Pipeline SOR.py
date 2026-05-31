import numpy as np
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import cKDTree

class BeadGeometryObserver:
    def __init__(self, params):
        self.p = params
        self.substrate_model = None

    def process_layer(self, layer_index, scan_profiles, current_z_shift):
        if not scan_profiles:
            return 0.0

        t_start = scan_profiles[0][2]
        valid_scans = []
        y_max_total = (scan_profiles[-1][2] - t_start) * self.p['PRINT_SPEED']

        for px, pz, ts in scan_profiles:
            y_calc = (ts - t_start) * self.p['PRINT_SPEED']
            if y_calc < self.p['TRIM_START'] or y_calc > (y_max_total - self.p['TRIM_END']):
                continue
            valid_scans.append((px, pz, y_calc))

        if not valid_scans:
            return 0.0

        if layer_index == 0:
            return self._build_substrate_model(valid_scans)
        else:
            return self._extract_layer_height(valid_scans, current_z_shift)

    def _build_substrate_model(self, valid_scans):
        all_x, all_y, all_z = [], [], []
        for px, pz, py in valid_scans:
            mask_x = (px >= self.p['X_MIN']) & (px <= self.p['X_MAX'])
            if np.any(mask_x):
                all_x.extend(px[mask_x])
                all_y.extend(np.full(np.sum(mask_x), py))
                all_z.extend(pz[mask_x])
        
        if not all_x: 
            return 0.0
        
        try:
            self.substrate_model = NearestNDInterpolator(list(zip(all_x, all_y)), all_z)
            return float(np.mean(all_z))
        except:
            return 0.0

    def _extract_layer_height(self, valid_scans, current_z_shift):
        if self.substrate_model is None:
            return 0.0
            
        all_x, all_y, all_h, all_sg = [], [], [], []

        for sg_idx, (px, pz, py) in enumerate(valid_scans):
            mask_roi = (px >= self.p['X_MIN']) & (px <= self.p['X_MAX'])
            if not np.any(mask_roi): continue
            
            rx = px[mask_roi]
            rz = pz[mask_roi]

            z_thresh = np.percentile(rz, self.p['PERC_MIN']) + (1.67 * self.p['ULS'])
            mask_z = rz <= z_thresh
            
            rx_f = rx[mask_z]
            rz_f = rz[mask_z]
            
            if len(rx_f) == 0: continue

            try:
                ref_z = self.substrate_model(rx_f, np.full(len(rx_f), py))
                h_local = current_z_shift + (ref_z - rz_f)
            except:
                continue

            all_x.extend(rx_f)
            all_y.extend(np.full(len(rx_f), py))
            all_h.extend(h_local)
            all_sg.extend(np.full(len(rx_f), sg_idx))

        if not all_x: return 0.0

        arr_x = np.array(all_x)
        arr_y = np.array(all_y)
        arr_h = np.array(all_h)
        arr_sg = np.array(all_sg)

        coords = np.column_stack([arr_x, arr_y, arr_h]).astype(np.float32)
        tree = cKDTree(coords)
        k_val = int(self.p['SOR_K']) + 1
        distances, _ = tree.query(coords, k=k_val)
        mean_dist = np.mean(distances[:, 1:], axis=1)
        
        global_mean = np.mean(mean_dist)
        global_std = np.std(mean_dist)
        threshold = global_mean + self.p['SOR_MULT'] * global_std
        mask_sor = mean_dist <= threshold

        valid_x = arr_x[mask_sor]
        valid_h = arr_h[mask_sor]
        valid_sg = arr_sg[mask_sor]

        if len(valid_x) == 0: return 0.0

        temp_apexes = []
        for sg in np.unique(valid_sg):
            mask = valid_sg == sg
            xg = valid_x[mask]
            hg = valid_h[mask]

            if len(xg) < self.p['MIN_PTS']: continue
            if xg.max() - xg.min() < self.p['MIN_WIDTH']: continue

            try:
                coeffs = np.polyfit(xg, hg, 2)
                a, b, c = coeffs
                
                if abs(a) > 1e-6:
                    vx = -b / (2*a)
                else:
                    vx = xg.mean()

                if not (self.p['X_MIN'] <= vx <= self.p['X_MAX']): continue

                vz = a*(vx**2) + b*vx + c

                if a >= -self.p['MIN_CURV']: continue

                temp_apexes.append({'x': vx, 'z': vz})
            except:
                continue

        if not temp_apexes: return 0.0

        apex_x = np.array([ap['x'] for ap in temp_apexes])
        apex_z = np.array([ap['z'] for ap in temp_apexes])

        med_apex_x = np.median(apex_x)
        mask_apex_tol = (apex_x >= med_apex_x - self.p['APEX_TOL']) & (apex_x <= med_apex_x + self.p['APEX_TOL'])
        final_apex_z = apex_z[mask_apex_tol]

        if len(final_apex_z) == 0: return 0.0

        return float(np.median(final_apex_z))