import numpy as np
import pandas as pd
import os
import sys

# Path setup to import SyInterferoPy
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SYINTERFEROPY_ROOT = os.path.join(PROJECT_ROOT, "SyInterferoPy")
if SYINTERFEROPY_ROOT not in sys.path:
    sys.path.insert(0, SYINTERFEROPY_ROOT)

from syinterferopy.syinterferopy import deformation_wrapper


# ---------- 1. Simple feature generation ----------

def compute_features(Ux, Uy, Uz, los):
    feats = {}

    # Max / min
    feats["Ux_max"] = float(np.max(Ux))
    feats["Uy_max"] = float(np.max(Uy))
    feats["Uz_max"] = float(np.max(Uz))

    feats["Ux_min"] = float(np.min(Ux))
    feats["Uy_min"] = float(np.min(Uy))
    feats["Uz_min"] = float(np.min(Uz))

    # Amplitudes
    feats["Ux_range"] = feats["Ux_max"] - feats["Ux_min"]
    feats["Uy_range"] = feats["Uy_max"] - feats["Uy_min"]
    feats["Uz_range"] = feats["Uz_max"] - feats["Uz_min"]

    # Simple ratios
    eps = 1e-6  # to avoid division by zero
    feats["Ux_Uz_ratio"] = feats["Ux_range"] / (feats["Uz_range"] + eps)
    feats["Uy_Uz_ratio"] = feats["Uy_range"] / (feats["Uz_range"] + eps)

    # Approximate "energy" on Uz
    feats["Uz_energy"] = float(np.sum(Uz**2))

    # LOS statistics
    feats["LOS_std"] = float(np.std(los))
    feats["LOS_max"] = float(np.max(los))

    return feats


# ---------- 2. Sampling physical parameters ----------

def sample_fault_normal():
    # Normal fault: shear, no opening
    params = {
        "strike": 0.0,
        "dip": float(np.random.uniform(40, 70)),               # dip angle α
        "length": float(np.random.uniform(3000, 8000)),        # in m
        "rake": -90.0,                                         # normal fault
        "slip": float(np.random.uniform(0.2, 1.0)),            # m
        "top_depth": float(np.random.uniform(500, 2000)),      # m
        "bottom_depth": float(np.random.uniform(3000, 6000)),  # m
        "opening": 0.0                                         # no opening
    }
    return params


def sample_intrusion_open():
    # Opening intrusion: opening, no shear
    params = {
        "strike": 0.0,
        "dip": float(np.random.uniform(70, 90)),               # almost vertical
        "length": float(np.random.uniform(3000, 8000)),
        "top_depth": float(np.random.uniform(500, 2000)),
        "bottom_depth": float(np.random.uniform(3000, 6000)),
        "opening": float(np.random.uniform(0.2, 1.0)),         # m
        "slip": 0.0,                                           # no shear
        "rake": -90.0                                          # value set but not really used if slip=0
    }
    return params


def sample_intrusion_sheared():
    # Sheared intrusion: opening + shear
    params = {
        "strike": 0.0,
        "dip": float(np.random.uniform(70, 90)),
        "length": float(np.random.uniform(3000, 8000)),
        "top_depth": float(np.random.uniform(500, 2000)),
        "bottom_depth": float(np.random.uniform(3000, 6000)),
        "opening": float(np.random.uniform(0.1, 0.7)),
        "slip": float(np.random.uniform(0.1, 0.7)),
        "rake": -90.0
    }
    return params


# ---------- 3. Main script ----------

def dataset_generator():
    np.random.seed(0)  # for reproducibility

    # Grid around Piton de la Fournaise (approx. coordinates)
    lon_c, lat_c = 55.708, -21.244
    n_pix = 128
    d_lon = 0.1
    d_lat = 0.1

    # Split interval [55.608, 55.808] into 128 points
    lons = np.linspace(lon_c - d_lon, lon_c + d_lon, n_pix)
    # Split interval [-21.344, -21.144] into 128 points
    lats = np.linspace(lat_c - d_lat, lat_c + d_lat, n_pix)
    # Build a 2D grid of coordinates from the vectors above
    lons_mg, lats_mg = np.meshgrid(lons, lats)

    dem = None  # flat topography for now
    deformation_ll = (lon_c, lat_c)  # center of the grid, where the source is located

    n_per_class = 50  # to increase later

    rows = []

    configs = [
        ("faille_normale", "quake", sample_fault_normal),
        ("intrusion_ouverte", "dyke", sample_intrusion_open),
        ("intrusion_cisaillee", "dyke", sample_intrusion_sheared),
    ]

    for label, source, sampler in configs:
        for _ in range(n_per_class):
            params = sampler()

            los, Ux, Uy, Uz = deformation_wrapper(
                lons_mg, lats_mg,
                deformation_ll,
                source,
                dem,
                **params
            )

            # Add simple Gaussian noise on LOS (optional)
            # noise_std = 0.005  # ~5 mm
            # los_noisy = los + np.random.normal(0, noise_std, size=los.shape)

            los_noisy = los  # no noise for now

            feats = compute_features(Ux, Uy, Uz, los_noisy)
            feats["label"] = label
            # Also keep some physical parameters (useful for analysis)
            for k, v in params.items():
                feats[f"phys_{k}"] = v

            rows.append(feats)

    df = pd.DataFrame(rows)
    output_dir = os.path.join(SCRIPT_DIR, "data")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "synthetic_faults_dataset.csv"), index=False)
    print(f"✅ Dataset sauvegardé : synthetic_faults_dataset.csv ({len(df)} lignes)")


if __name__ == "__main__":
    dataset_generator()
