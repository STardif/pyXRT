"""
Module: bliss_hdf5_reader.py

Description:
    Provides a robust, generic reader for BLISS HDF5 files. 
    It features lazy initialization, dynamic inspection of available 
    counters in the 'measurement' group, and safe 2D grid reshaping.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Any

import h5py
import numpy as np

# Configure basic logging for verbose output
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class BlissMeshScan:
    """
    A class to read, inspect, and reshape 2D mesh scans from a BLISS HDF5 file.
    """

    def __init__(
        self,
        h5_filepath: Union[str, Path],
        scan_number: int,
        image_prefix: str = "eiger",
        image_suffix: str = ".h5",
        verbose: bool = False
    ):
        self.h5_filepath = Path(h5_filepath)
        self.scan_number = scan_number
        self.scan_key = f"{scan_number}.1"
        self.image_prefix = image_prefix
        self.image_suffix = image_suffix
        self.verbose = verbose

        self.counters: Dict[str, np.ndarray] = {}
        self.grids: Dict[str, np.ndarray] = {}
        self.image_path_list: List[Path] = []
        self.nb_images: int = 0
        
        # Extract basic metadata upon instantiation (lightweight)
        self._read_metadata()

    def _read_metadata(self) -> None:
        """Opens the HDF5 file to extract the title and define the scan geometry."""
        if not self.h5_filepath.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_filepath}")

        with h5py.File(self.h5_filepath, "r") as f:
            if self.scan_key not in f:
                raise KeyError(f"Scan {self.scan_key} not found in {self.h5_filepath.name}")

            scan_group = f[self.scan_key]
            self.title = scan_group["title"][()].decode("utf-8")
            
            # Extract fast and slow motor names directly from the title string
            title_parts = self.title.split()
            if len(title_parts) >= 10 and title_parts[0] in ("amesh", "dmesh"):
                self.scan_type = title_parts[0]
                
                self.fast_motor_name = title_parts[1]
                self.fast_start = float(title_parts[2])
                self.fast_stop = float(title_parts[3])
                self.fast_intervals = int(title_parts[4])
                self.fast_steps = self.fast_intervals + 1
                self.fast_step_size = (self.fast_stop - self.fast_start) / self.fast_intervals if self.fast_intervals else 0.0

                self.slow_motor_name = title_parts[5]
                self.slow_start = float(title_parts[6])
                self.slow_stop = float(title_parts[7])
                self.slow_intervals = int(title_parts[8])
                self.slow_steps = self.slow_intervals + 1
                self.slow_step_size = (self.slow_stop - self.slow_start) / self.slow_intervals if self.slow_intervals else 0.0

                self.count_time = float(title_parts[9])
                
                # Matrix shape (rows, cols) -> (slow, fast)
                self.scan_shape = (self.slow_steps, self.fast_steps)
                self.expected_points = self.slow_steps * self.fast_steps
            else:
                raise ValueError(f"Scan {self.scan_key} is not a recognized 2D mesh scan. Title: {self.title}")

    def get_available_counters(self) -> Dict[str, Dict[str, Any]]:
        """
        Dynamically inspects the 'measurement' group and returns metadata 
        for all available counters without loading them into memory.
        
        Returns:
            Dict mapping counter names to a dictionary containing 'shape' and 'dtype'.
        """
        counters_info = {}
        with h5py.File(self.h5_filepath, "r") as f:
            meas_group = f[f"{self.scan_key}/measurement"]
            for key in meas_group.keys():
                dataset = meas_group[key]
                if isinstance(dataset, h5py.Dataset):
                    counters_info[key] = {
                        "shape": dataset.shape,
                        "dtype": str(dataset.dtype)
                    }
        return counters_info

    def read_data(self, extra_counters: Optional[List[str]] = None) -> None:
        """
        Loads the core motors and requested extra counters into memory, 
        locates images, and reshapes the 1D arrays into 2D grids.
        
        Args:
            extra_counters: List of specific counter names to load from the HDF5 file.
        """
        extra_counters = extra_counters or []
        
        with h5py.File(self.h5_filepath, "r") as f:
            meas_group = f[f"{self.scan_key}/measurement"]
            
            # Read core motors
            for motor in (self.fast_motor_name, self.slow_motor_name):
                if motor in meas_group:
                    self.counters[motor] = meas_group[motor][()]
                else:
                    logging.warning(f"Core motor '{motor}' not found in measurement group.")

            # Read explicitly requested extra counters
            for counter in extra_counters:
                if counter in meas_group:
                    self.counters[counter] = meas_group[counter][()]
                else:
                    logging.warning(f"Extra counter '{counter}' not found in measurement group.")

        self._process_images()
        self._reshape_grids()
        
        if self.verbose:
            logging.info(self.get_summary())

    def _process_images(self) -> None:
        """Locates and safely truncates the image path list."""
        self.scan_folder = self.h5_filepath.parent / f"scan{self.scan_number:04d}"
        
        if not self.scan_folder.exists():
            logging.warning(f"Image folder not found: {self.scan_folder}")
            return

        self.image_path_list = sorted(self.scan_folder.rglob(f"{self.image_prefix}*{self.image_suffix}"))
        
        # Safely truncate the list to the expected number of points
        self.image_path_list = self.image_path_list[:self.expected_points]
        self.nb_images = len(self.image_path_list)

    def _reshape_grids(self) -> None:
        """
        Reshapes 1D BLISS datasets into 2D grids mapping the physical scan.
        Handles aborted scans by padding with NaNs.
        """
        for name, data_1d in self.counters.items():
            actual_len = data_1d.size
            
            # We only reshape 1D counters (ignore 2D spectra if mistakenly loaded)
            if data_1d.ndim != 1:
                logging.warning(f"Skipping reshape for '{name}'; expected 1D array but got {data_1d.ndim}D.")
                continue

            if actual_len == self.expected_points:
                grid = data_1d.reshape(self.scan_shape)
            else:
                # Handle aborted scan: pad with NaNs
                logging.warning(f"Aborted scan detected for '{name}'. Padding missing points with NaN.")
                padded_data = np.full(self.expected_points, np.nan, dtype=data_1d.dtype)
                padded_data[:actual_len] = data_1d
                grid = padded_data.reshape(self.scan_shape)
                
            # Preserve original axis swapping if Y is the fast motor
            if self.fast_motor_name == "yech":
                grid = np.swapaxes(grid, 0, 1)
                
            self.grids[name] = grid

    def get_relative_position_grid(self, motor_name: str) -> np.ndarray:
        """Returns a 2D grid of motor positions relative to the start point (in microns)."""
        if motor_name not in self.grids:
            raise KeyError(f"Motor '{motor_name}' data not available in grids.")
        
        grid = self.grids[motor_name]
        origin = np.nanmin(grid) if np.isnan(grid[0, 0]) else grid[0, 0]
        return (grid - origin) * 1000.0

    def get_summary(self) -> str:
        """Generates a summary string of the scan parameters."""
        info = [
            f"Scan Title: {self.title}",
            f"Mesh Shape: {self.scan_shape} (Slow x Fast)",
            f"Fast Motor ({self.fast_motor_name}): Range {(self.fast_stop - self.fast_start)*1000:.3f} µm, Step {self.fast_step_size*1000:.3f} µm",
            f"Slow Motor ({self.slow_motor_name}): Range {(self.slow_stop - self.slow_start)*1000:.3f} µm, Step {self.slow_step_size*1000:.3f} µm",
            f"Expected Images: {self.expected_points}",
            f"Found Images ({self.image_prefix}*{self.image_suffix} files): {self.nb_images}",
            f"Loaded Grids: {list(self.grids.keys())}"
        ]
        return "\n".join(info)


def main() -> None:
    """Example execution demonstrating the usage of BlissMeshScan."""
    h5_file = "/data/visitor/ihma907/bm32/20260909/RAW_DATA/epiGan_hex_v2/epiGan_hex_v2_hex8/epiGan_hex_v2_hex8.h5"
    
    try:
        # 1. Lightweight instantiation
        scan = BlissMeshScan(h5_filepath=h5_file, scan_number=1, verbose=True)
        scan.read_data()
        #print(scan.get_summary())
        
        # 2. Inspect available counters dynamically
        print("\n--- Inspecting Available Counters ---")
        available_counters = scan.get_available_counters()
        for name, info in available_counters.items():
            print(f"Counter: {name:<20} Shape: {str(info['shape']):<12} Dtype: {info['dtype']}")
            
        # 3. Read specific data based on inspection
        #requested_counters = ["epoch", "mon"]
        #print(f"\n--- Loading Data for core motors and: {requested_counters} ---")
        #scan.read_data(extra_counters=requested_counters)
        
        # 4. Use the parsed grids
        #if scan.fast_motor_name in scan.grids:
        #    dx_um = scan.get_relative_position_grid(scan.fast_motor_name)
        #    print(f"\n{scan.fast_motor_name} relative range: {np.nanmax(dx_um):.2f} µm")

    except Exception as e:
        logging.error(f"Execution failed: {e}")


if __name__ == "__main__":
    main()