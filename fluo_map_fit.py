# Authors: Gemini and Samuel Tardif

"""
Module: constrained_xrf_fitter.py

Description:
    Reads X-ray fluorescence spectra from a BLISS HDF5 file and performs
    parallelized Gaussian peak fitting. Constrains the K_alpha and K_beta 
    lines for each element to fixed relative positions and amplitude ratios
    to stabilize the fit across a 2D spatial map.
"""

import logging
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Optional

import h5py
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# =============================================================================
# Physics & Detector Configuration
# At ~10 eV/channel, Ga Ka is ~9.25 keV (ch 925), Ga Kb is ~10.26 keV (ch 1026).
# Delta is ~101 channels. Typical Kb/Ka branching ratio is ~0.13.
# Update the Indium values based on the specific L-lines or K-lines in your ROI.
# =============================================================================
GA_DELTA_CH = 101.0       # Channels between Ga Ka and Ga Kb
GA_KB_RATIO = 0.13        # Amplitude ratio of Ga Kb to Ga Ka

IN_DELTA_CH = 45.0        # Placeholder: Update with exact delta for In lines in ROI
IN_KB_RATIO = 0.15        # Placeholder: Update with exact branching ratio for In

def coupled_double_gaussian(
    x: np.ndarray, 
    amp_alpha: float, 
    cen_alpha: float, 
    sigma: float, 
    delta: float, 
    ratio: float
) -> np.ndarray:
    """
    Generates a double Gaussian profile where the secondary peak is rigidly 
    constrained relative to the primary peak.
    """
    primary = amp_alpha * np.exp(-((x - cen_alpha) ** 2) / (2 * sigma ** 2))
    secondary = (amp_alpha * ratio) * np.exp(-((x - (cen_alpha + delta)) ** 2) / (2 * sigma ** 2))
    return primary + secondary

def spectrum_model(
    x: np.ndarray, 
    bkg: float, 
    amp_ga: float, 
    cen_ga: float, 
    sig_ga: float, 
    amp_in: float, 
    cen_in: float, 
    sig_in: float
) -> np.ndarray:
    """
    Complete model comprising a constant background and two constrained elements.
    """
    ga_profile = coupled_double_gaussian(x, amp_ga, cen_ga, sig_ga, GA_DELTA_CH, GA_KB_RATIO)
    in_profile = coupled_double_gaussian(x, amp_in, cen_in, sig_in, IN_DELTA_CH, IN_KB_RATIO)
    return bkg + ga_profile + in_profile

def fit_single_spectrum(args: Tuple[int, np.ndarray, np.ndarray]) -> Tuple[int, Optional[float], Optional[float]]:
    """
    Fits a single XRF spectrum. Designed to be mapped across a ProcessPool.
    
    Args:
        args: A tuple containing (pixel_index, channel_array, spectrum_counts).
        
    Returns:
        tuple: (pixel_index, area_ga, area_in). Returns None for areas if the fit fails.
    """
    index, x_data, y_data = args
    
    # p0: [bkg, amp_ga, cen_ga, sig_ga, amp_in, cen_in, sig_in]
    # Adjust starting centers based on your exact ROI calibration
    p0 = [np.min(y_data), np.max(y_data)*0.5, 925, 10, np.max(y_data)*0.5, 1100, 10]
    
    # Restrict centers from wandering too far from theoretical positions
    bounds = (
        [0, 0, 915, 1, 0, 1050, 1],
        [np.inf, np.inf, 935, 30, np.inf, 1150, 30]
    )
    
    try:
        popt, _ = curve_fit(spectrum_model, x_data, y_data, p0=p0, bounds=bounds)
        
        # Calculate total elemental area (Integral of Ka + Integral of Kb)
        # Area = amp * sigma * sqrt(2*pi)
        ga_area_alpha = popt[1] * popt[3] * np.sqrt(2 * np.pi)
        ga_area_total = ga_area_alpha + (ga_area_alpha * GA_KB_RATIO)
        
        in_area_alpha = popt[4] * popt[6] * np.sqrt(2 * np.pi)
        in_area_total = in_area_alpha + (in_area_alpha * IN_KB_RATIO)
        
        return index, ga_area_total, in_area_total
        
    except RuntimeError:
        return index, None, None

def process_and_plot(h5_filepath: str, dataset_path: str, scan_shape: Tuple[int, int]) -> None:
    """Loads data, manages the multiprocessing pool, and plots results."""
    h5_path = Path(h5_filepath)
    
    logging.info(f"Loading dataset {dataset_path} from {h5_path.name}...")
    with h5py.File(h5_path, "r") as f:
        raw_spectra = f[dataset_path][()]
        
    n_pixels = raw_spectra.shape[0]
    roi_start, roi_end = 800, 1330
    x_data = np.arange(roi_start, roi_end)
    
    # Slice the ROI prior to multiprocessing to minimize serialization payloads
    tasks = [
        (i, x_data, raw_spectra[i, roi_start:roi_end]) 
        for i in range(n_pixels)
    ]
    
    # Reserve one core for OS/scheduler overhead
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    logging.info(f"Initiating ProcessPoolExecutor with {max_workers} workers for {n_pixels} spectra...")
    
    ga_areas = np.full(n_pixels, np.nan)
    in_areas = np.full(n_pixels, np.nan)
    
    # Execute fits in parallel with chunking for IPC efficiency
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(fit_single_spectrum, tasks, chunksize=100)
        
        for idx, ga_area, in_area in results:
            if ga_area is not None:
                ga_areas[idx] = ga_area
                in_areas[idx] = in_area

    map_ga = ga_areas.reshape(scan_shape)
    map_in = in_areas.reshape(scan_shape)
    
    logging.info("Fitting complete. Rendering plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = ax1.imshow(map_ga, aspect='auto', cmap='plasma')
    ax1.set_title("Gallium Distribution (Total Area)")
    fig.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(map_in, aspect='auto', cmap='plasma')
    ax2.set_title("Indium Distribution (Total Area)")
    fig.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define file and parameters
    H5_FILE = "InPoSi_v2_center.h5"
    DATASET_PATH = "1.1/measurement/xia_det0"
    
    # Update grid dimensions based on the amesh metadata title
    SCAN_SHAPE = (81, 81) 
    
    process_and_plot(H5_FILE, DATASET_PATH, SCAN_SHAPE)