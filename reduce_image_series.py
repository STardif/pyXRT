"""
Authors: Gemini and Samuel Tardif
Module: reduce_image_series.py

Description:
    Performs memory-efficient, highly parallelized pixel-wise min, max, 
    and mean reductions across independent HDF5 image files from a BLISS scan.
    Integrates with bliss_hdf5_reader_2 to automatically discover frames 
    associated with a specific scan number.
    Usage:
    python reduce_image_series.py /data/visitor/ihma907/bm32/20260909/RAW_DATA/epiGan_hex_v2/epiGan_hex_v2_hex8/epiGan_hex_v2_hex8.h5 -s 1 -o /data/bm32/inhouse/STAFF/SAM/ihma907/bm32/20260909/PROCESSED_DATA/
"""

import argparse
import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm

# Import the BLISS scan reader
from bliss_hdf5_reader import BlissMeshScan

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute pixel-wise min, max, and mean reductions for a BLISS scan series."
    )
    parser.add_argument(
        "h5_file",
        type=Path,
        help="Path to the master/dataset BLISS HDF5 file (e.g., epiGan_hex_v2_hex8.h5)."
    )
    parser.add_argument(
        "-s", "--scan",
        type=int,
        required=True,
        help="Scan number to process (e.g., 1 for '1.1')."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=Path,
        required=True,
        help="Output directory to save the reduced image HDF5 file."
    )
    parser.add_argument(
        "--image_suffix",
        type=str,
        default=".h5",
        help="File extension of the detector frames (default: .h5)."
    )
    parser.add_argument(
        "-c", "--chunk_size",
        type=int,
        default=50,
        help="Number of image files per worker chunk (default: 50)."
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="Number of CPU cores to utilize (default: all available cores)."
    )
    return parser.parse_args()


def _read_frame_array(filepath: Path) -> np.ndarray:
    """
    Safely extracts the 2D image array from an independent HDF5 detector file.
    Uses fast direct lookup on common paths before falling back to recursive search.
    """
    with h5py.File(filepath, "r") as h5f:
        common_paths = [
            "/entry_0000/measurement/data",
            "/entry_0000/instrument/eiger4m/data",
            "/entry_0000/CRGIF/eiger4m/data",
        ]
        for path in common_paths:
            if path in h5f:
                ds = h5f[path]
                return ds[0] if ds.ndim == 3 else ds[()]

        # Fallback: traverse tree to find the first 2D/3D dataset
        found_path: List[str] = []

        def _visitor(name: str, obj: h5py.HLObject) -> Optional[bool]:
            if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                found_path.append(obj.name)
                return True
            return None

        entry = h5f.get("/entry_0000")
        if entry:
            entry.visititems(_visitor)
            if found_path:
                ds = h5f[found_path[0]]
                return ds[0] if ds.ndim == 3 else ds[()]

    raise KeyError(f"Could not locate a valid image dataset inside {filepath.name}")


def _worker_process_files(file_paths: List[Path]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Map Worker: Reads a chunk of HDF5 files and computes local min, max, and sum.

    Args:
        file_paths: List of file paths to process within this worker.

    Returns:
        Tuple of (local_min, local_max, local_sum, valid_frames_count).
    """
    local_min = None
    local_max = None
    local_sum = None
    valid_frames = 0

    for file_path in file_paths:
        try:
            data = _read_frame_array(file_path)
            data = np.squeeze(data)

            if local_min is None:
                local_min = data.copy()
                local_max = data.copy()
                # Cast to float64 to prevent integer overflow during multi-thousand image sums
                local_sum = data.astype(np.float64)
            else:
                local_min = np.minimum(local_min, data)
                local_max = np.maximum(local_max, data)
                local_sum += data

            valid_frames += 1

        except Exception as err:
            logging.error(f"Error processing {file_path.name}: {err}. Skipping.")
            continue

    return local_min, local_max, local_sum, valid_frames


def compute_scan_reductions(
    image_files: List[Path],
    chunk_size: int = 50,
    num_workers: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Orchestrates the parallel MapReduce pipeline across all resolved image files.
    """
    total_files = len(image_files)
    if total_files == 0:
        raise ValueError("Image file list is empty.")

    # Partition file list into chunks
    tasks = [
        image_files[i : i + chunk_size]
        for i in range(0, total_files, chunk_size)
    ]

    workers = num_workers or mp.cpu_count()
    logging.info(f"Distributing {total_files} files across {workers} workers (chunk size: {chunk_size})...")

    global_min = None
    global_max = None
    global_sum = None
    total_processed = 0

    with mp.Pool(processes=workers) as pool:
        for loc_min, loc_max, loc_sum, count in tqdm(
            pool.imap_unordered(_worker_process_files, tasks),
            total=len(tasks),
            desc="Processing chunks"
        ):
            if count == 0 or loc_min is None:
                continue

            if global_min is None:
                global_min = loc_min
                global_max = loc_max
                global_sum = loc_sum
            else:
                global_min = np.minimum(global_min, loc_min)
                global_max = np.maximum(global_max, loc_max)
                global_sum += loc_sum

            total_processed += count

    if total_processed == 0 or global_sum is None:
        raise RuntimeError("No valid frames could be processed. Please check file integrity.")

    logging.info(f"Reduction complete. Successfully processed {total_processed}/{total_files} frames.")
    global_mean = global_sum / total_processed

    return global_min, global_max, global_mean


def save_reduced_hdf5(
    output_path: Path,
    global_min: np.ndarray,
    global_max: np.ndarray,
    global_mean: np.ndarray,
    metadata: dict
) -> None:
    """Saves the reduced 2D arrays to an HDF5 file with GZIP compression."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving reduced datasets to {output_path}...")

    with h5py.File(output_path, "w") as h5f:
        # Save datasets with GZIP compression level 4 (optimal speed vs compression ratio)
        h5f.create_dataset("min_image", data=global_min, compression="gzip", compression_opts=4)
        h5f.create_dataset("max_image", data=global_max, compression="gzip", compression_opts=4)
        h5f.create_dataset("mean_image", data=global_mean, compression="gzip", compression_opts=4)

        # Store contextual metadata as HDF5 root attributes
        for key, val in metadata.items():
            h5f.attrs[key] = str(val)

    logging.info("Save successful.")


def main() -> None:
    args = parse_args()

    if not args.h5_file.exists():
        raise FileNotFoundError(f"Master HDF5 file not found: {args.h5_file}")

    logging.info(f"Initializing scan {args.scan} from {args.h5_file.name}...")
    scan = BlissMeshScan(
        h5_filepath=args.h5_file,
        scan_number=args.scan,
        image_suffix=args.image_suffix
    )
    scan.read_data()

    if scan.nb_images == 0:
        raise RuntimeError(f"No image files matching '*{args.image_suffix}' found in {scan.scan_folder}")

    # Compute global reductions
    g_min, g_max, g_mean = compute_scan_reductions(
        image_files=scan.image_path_list,
        chunk_size=args.chunk_size,
        num_workers=args.workers
    )

    # Save to output directory
    output_filename = f"{args.h5_file.stem}_scan{args.scan:04d}_reduced.h5"
    output_filepath = args.output_dir / output_filename

    metadata = {
        "master_h5": str(args.h5_file.resolve()),
        "scan_number": args.scan,
        "scan_title": scan.title,
        "processed_frames": scan.nb_images,
    }

    save_reduced_hdf5(output_filepath, g_min, g_max, g_mean, metadata)


if __name__ == "__main__":
    main()