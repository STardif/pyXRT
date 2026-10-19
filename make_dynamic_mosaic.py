"""
Module: make_dynamic_mosaic.py

Description:
    Production-ready pipeline to process X-ray diffraction frames from BM32.
    Accepts a master HDF5 file, scan number, and ROI parameters via the CLI.
    Extracts physical motor coordinates directly from the BLISS HDF5 structure
    using bliss_hdf5_reader. Computes spatial metrics (Entropy, Max-to-Mean, 
    CoM displacement), stitches a full 2D mosaic, and generates a summary 
    figure mapped to the physical motor axes.
"""

import argparse
import multiprocessing as mp
from pathlib import Path
from functools import partial
from typing import Dict, Tuple, List

import h5py
import fabio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import the robust reader class provided by the user
from bliss_hdf5_reader import BlissMeshScan


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for dynamic ROI and scan selection."""
    parser = argparse.ArgumentParser(
        description="Process X-ray diffraction frames and compute spatial metrics using physical motor coordinates."
    )
    parser.add_argument("h5_file", type=Path, help="Path to the master BLISS HDF5 file.")
    parser.add_argument("-s", "--scan", type=int, required=True, help="Scan number to process (e.g., 1 for '1.1').")
    parser.add_argument("-rx", "--roi_x", type=int, required=True, help="Center X coordinate of the ROI.")
    parser.add_argument("-ry", "--roi_y", type=int, required=True, help="Center Y coordinate of the ROI.")
    parser.add_argument("-w", "--hws", type=int, default=5, help="Half-window size for the ROI (default: 5).")
    parser.add_argument("--image_suffix", type=str, default=".h5", help="Suffix of the image files (default: .h5).")
    parser.add_argument("-o", "--output_dir", type=Path, default=".", help="Output directory for the data.")
    return parser.parse_args()


def read_image(filepath: Path) -> np.ndarray:
    """Reads HDF5 or TIFF detector frames."""
    if filepath.suffix == ".h5":
        with h5py.File(filepath, 'r') as h5f:
            # Corrected internal HDF5 path based on eiger4m_0000.h5 structure
            return h5f['/entry_0000/CRGIF/eiger4m/data'][:][0]
    elif filepath.suffix in (".tif", ".tiff"):
        with fabio.open(filepath) as im:
            return im.data
    raise ValueError(f"Unsupported file format: {filepath}")


def roi_slices(roi_x: int, roi_y: int, hws: int) -> Tuple[slice, slice]:
    """(row, column) slices of the ROI in a detector frame."""
    return (slice(roi_y - hws, roi_y + hws + 1), slice(roi_x - hws, roi_x + hws + 1))


def get_image_roi(filepath: Path, roi_x: int, roi_y: int, hws: int) -> np.ndarray:
    """Extracts a Region of Interest from an image file (one frame per file).
    Only used when the master file gives no direct access to the frames."""
    data = read_image(filepath)
    return data[roi_slices(roi_x, roi_y, hws)]


# Set once in each worker process, so the scan is not sent with every task.
_WORKER_SCAN = None


def _init_worker(scan: BlissMeshScan) -> None:
    global _WORKER_SCAN
    _WORKER_SCAN = scan


def _read_roi_batch(indices: List[int], roi: Tuple[slice, slice]) -> List[np.ndarray]:
    """ROIs of a batch of frames, read through the master file opened once."""
    return list(_WORKER_SCAN.iter_frames(indices=indices, roi=roi))


def read_rois(scan: BlissMeshScan, roi: Tuple[slice, slice], batch: int = 50) -> List[np.ndarray]:
    """ROI of every recorded frame, in acquisition order, read in parallel."""
    n = scan.nb_frames
    batches = [list(range(i, min(i + batch, n))) for i in range(0, n, batch)]
    rois: List[np.ndarray] = []
    with mp.Pool(min(mp.cpu_count(), len(batches)), initializer=_init_worker,
                 initargs=(scan,)) as pool:
        with tqdm(total=n) as bar:
            for out in pool.imap(partial(_read_roi_batch, roi=roi), batches):
                rois.extend(out)
                bar.update(len(out))
    return rois


def fill_missing_positions(grid: np.ndarray) -> np.ndarray:
    """Motor positions of points never recorded (aborted scan) are NaN, which
    pcolormesh/quiver refuse. Fill them from a plane fitted to the recorded
    positions (a mesh is linear in row and column index)."""
    missing = ~np.isfinite(grid)
    if not missing.any():
        return grid
    rows, cols = np.indices(grid.shape)
    design = np.column_stack([np.ones(grid.size), rows.ravel(), cols.ravel()])
    ok = ~missing.ravel()
    coef, *_ = np.linalg.lstsq(design[ok], grid.ravel()[ok], rcond=None)
    filled = grid.copy()
    filled[missing] = (design @ coef).reshape(grid.shape)[missing]
    return filled


def to_scan_grid(rois: List[np.ndarray], index_grid: np.ndarray) -> List[np.ndarray]:
    """Order the ROIs like the scan grid (row-major), using the scan's index grid.
    Handles back-and-forth meshes; points never recorded (aborted scan) become
    empty (zero) blocks."""
    empty = np.zeros_like(rois[0])
    return [rois[i] if 0 <= i < len(rois) else empty for i in index_grid.ravel()]


def compute_pipeline_data(
    array_list: List[np.ndarray], grid_shape: Tuple[int, int]
) -> Dict[str, np.ndarray]:
    """Consolidated computation for metrics, mosaic stitching, and extreme ROI extraction."""
    grid_rows, grid_cols = grid_shape
    stacked_3d = np.array(array_list, dtype=np.float64)
    N, block_rows, block_cols = stacked_3d.shape
    flat_blocks = stacked_3d.reshape(N, -1)

    # 1. Stitch Mosaic
    grid_4d = stacked_3d.reshape(grid_rows, grid_cols, block_rows, block_cols)
    transposed_4d = grid_4d.transpose(0, 2, 1, 3)
    mosaic = transposed_4d.reshape(grid_rows * block_rows, grid_cols * block_cols)

    # 2. Shannon Entropy & Extremes
    block_sums = flat_blocks.sum(axis=1, keepdims=True)
    safe_sums = np.where(block_sums == 0, 1.0, block_sums)
    p = flat_blocks / safe_sums
    p_safe = np.where(p <= 0, 1e-12, p)
    entropy = -np.sum(p * np.log(p_safe), axis=1)
    entropy[block_sums.flatten() == 0] = 0.0
    
    # Ignore empty frames (entropy == 0.0) when finding the sharpest peak
    valid_entropy = np.where(entropy > 0, entropy, np.inf)
    sharpest_idx = np.argmin(valid_entropy)
    diffuse_idx = np.argmax(entropy)

    # 3. Max-to-Mean
    block_maxes = flat_blocks.max(axis=1)
    block_means = flat_blocks.mean(axis=1)
    safe_means = np.where(block_means == 0, 1.0, block_means)
    max_to_mean = block_maxes / safe_means
    max_to_mean[block_means == 0] = 0.0

    # Total intensity
    total_intensity = block_sums.reshape(grid_rows, grid_cols)
    
    # 4. Center of Mass Vector Field
    total_mass = stacked_3d.sum(axis=(1, 2))
    safe_mass = np.where(total_mass == 0, 1.0, total_mass)
    
    y_coords = np.arange(block_rows).reshape(block_rows, 1)
    x_coords = np.arange(block_cols).reshape(1, block_cols)
    
    x_com = (stacked_3d * x_coords).sum(axis=(1, 2)) / safe_mass
    y_com = (stacked_3d * y_coords).sum(axis=(1, 2)) / safe_mass
    
    # Calculate displacement relative to the global mean CoM position
    dx = x_com - np.mean(x_com[total_mass > 0])
    dy = y_com - np.mean(y_com[total_mass > 0])
    
    # Force empty blocks to have zero displacement
    dx[total_mass == 0] = 0.0
    dy[total_mass == 0] = 0.0

    return {
        "mosaic": mosaic,
        "entropy": entropy.reshape(grid_rows, grid_cols),
        "max_to_mean": max_to_mean.reshape(grid_rows, grid_cols),
        "com_dx": dx.reshape(grid_rows, grid_cols),
        "com_dy": dy.reshape(grid_rows, grid_cols),
        "sharpest_roi": array_list[sharpest_idx],
        "diffuse_roi": array_list[diffuse_idx],
        "total_intensity": total_intensity
    }


def save_metrics_to_hdf5(data: Dict[str, np.ndarray], output_path: Path) -> None:
    """Saves computed arrays to an HDF5 file with GZIP compression."""
    with h5py.File(output_path, 'w') as h5f:
        for key, array_data in data.items():
            h5f.create_dataset(
                key, 
                data=array_data, 
                compression="gzip", 
                compression_opts=4
            )
    print(f"Data successfully saved to {output_path}")


def plot_and_save_figure(
    data: Dict[str, np.ndarray], 
    x_grid: np.ndarray, 
    y_grid: np.ndarray, 
    mon_grid: np.ndarray, 
    motor_names: Tuple[str, str], 
    output_path: Path
) -> None:
    """Generates and saves a comprehensive summary figure mapped to physical coordinates."""
    fig = plt.figure(figsize=(18, 12))
    
    x_name, y_name = motor_names
    x_min, x_max = np.nanmin(x_grid), np.nanmax(x_grid)
    y_min, y_max = np.nanmin(y_grid), np.nanmax(y_grid)
    
    # Extent bounds for the full mosaic image to map it to physical dimensions
    extent = [x_min, x_max, y_max, y_min]

    ax_mosaic = plt.subplot(2, 3, 1)
    ax_sharp = plt.subplot(2, 3, 2)
    ax_diffuse = plt.subplot(2, 3, 3)
    ax_entropy = plt.subplot(2, 3, 4)
    ax_papr = plt.subplot(2, 3, 5)
    ax_quiver = plt.subplot(2, 3, 6)

    # 1. Full Mosaic (Plotted with physical extent)
    im_mos = ax_mosaic.imshow(data["mosaic"], cmap='bone', extent=extent, aspect='auto')
    ax_mosaic.set_title('Full Stitched Mosaic')
    ax_mosaic.set_xlabel(f"{x_name} (Physical)")
    ax_mosaic.set_ylabel(f"{y_name} (Physical)")
    ax_mosaic.invert_yaxis()  # Standardize orientation
    ax_mosaic.set_aspect("equal")
    fig.colorbar(im_mos, ax=ax_mosaic, fraction=0.046, pad=0.04)

    # 2. Sharpest ROI (Pixel coordinates are used as it is a single isolated block)
    im_sharp = ax_sharp.imshow(data["sharpest_roi"], cmap='viridis')
    ax_sharp.set_title('Sharpest Peak (Min Entropy)')
    fig.colorbar(im_sharp, ax=ax_sharp, fraction=0.046, pad=0.04)

    # 3. Diffuse ROI
    im_diff = ax_diffuse.imshow(data["diffuse_roi"], cmap='viridis')
    ax_diffuse.set_title('Most Diffuse Block (Max Entropy)')
    fig.colorbar(im_diff, ax=ax_diffuse, fraction=0.046, pad=0.04)

    # 4. Entropy Grid (Mapped via pcolormesh for exact physical alignment)
    im_ent = ax_entropy.pcolormesh(x_grid, y_grid, data["entropy"], cmap='viridis_r', shading='nearest')
    ax_entropy.set_title('Entropy Grid Map\n(Dark/Low = Sharper)')
    ax_entropy.set_xlabel(x_name)
    ax_entropy.set_ylabel(y_name)
    fig.colorbar(im_ent, ax=ax_entropy, fraction=0.046, pad=0.04)

    # 5. Total intensity
    # Prevent division by zero if monitor grid has zeros
    safe_mon_grid = np.where(mon_grid == 0, 1.0, mon_grid)
    im_papr = ax_papr.pcolormesh(x_grid, y_grid, data["total_intensity"]/safe_mon_grid, cmap='magma', shading='nearest')
    ax_papr.set_title('Total intensity (normalized by monitor)')
    ax_papr.set_xlabel(x_name)
    ax_papr.set_ylabel(y_name)
    fig.colorbar(im_papr, ax=ax_papr, fraction=0.046, pad=0.04)

    # 6. CoM Quiver Plot (Vector field representing shift from the mean)
    rows, cols = data["com_dx"].shape
    stride = 2 if cols >= 50 else 1  # Prevents arrow overcrowding on dense grids
    
    ax_quiver.quiver(
        x_grid[::stride, ::stride], 
        y_grid[::stride, ::stride], 
        data["com_dx"][::stride, ::stride], 
        data["com_dy"][::stride, ::stride], 
        color='black',
        angles='xy'
    )
    ax_quiver.set_title('CoM Displacement Vector Field\n(Relative to Mean)')
    ax_quiver.set_xlabel(x_name)
    ax_quiver.set_ylabel(y_name)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Summary figure successfully saved to {output_path}")
    plt.close()


def main() -> None:
    args = parse_args()

    if not args.h5_file.exists():
        raise FileNotFoundError(f"Master HDF5 file not found: {args.h5_file}")

    print(f"Reading scan {args.scan} metadata from {args.h5_file.name}...")
    scan = BlissMeshScan(
        h5_filepath=args.h5_file, 
        scan_number=args.scan, 
        image_suffix=args.image_suffix
    )
    scan.read_data(extra_counters=["mon",])
    print(scan.summary())

    if scan.nb_frames == 0:
        raise RuntimeError(f"No images found for scan {args.scan} in {scan.scan_folder}")

    # Determine which motors to use for X and Y plotting based on names
    x_motor = next((m for m in (scan.fast_motor_name, scan.slow_motor_name) if 'x' in m.lower()),
                   scan.fast_motor_name)
    y_motor = scan.slow_motor_name if x_motor == scan.fast_motor_name else scan.fast_motor_name

    x_grid = fill_missing_positions(scan.grids[x_motor])
    y_grid = fill_missing_positions(scan.grids[y_motor])

    # Attempt to use the actual monitor grid; fallback to ones if missing
    if "mon" in scan.grids:
        mon_grid = scan.grids["mon"]
    else:
        print("Warning: 'mon' counter not found. Falling back to unnormalized intensity.")
        mon_grid = np.ones_like(x_grid)

    print(f"Extracting ROIs in parallel from {scan.nb_frames} frames...")
    roi = roi_slices(args.roi_x, args.roi_y, args.hws)
    if scan.frame_sources():
        rois = read_rois(scan, roi)
    else:
        # No virtual dataset in the master file (e.g. TIFF images): one file per frame.
        worker_func = partial(get_image_roi, roi_x=args.roi_x, roi_y=args.roi_y, hws=args.hws)
        with mp.Pool(mp.cpu_count()) as pool:
            rois = list(tqdm(pool.imap(worker_func, scan.image_path_list, chunksize=50),
                             total=scan.nb_images))

    # Blocks in grid order: rows follow the slow motor, columns the fast one.
    blocks = to_scan_grid(rois, scan.index_grid)
    grid_shape = scan.scan_shape
    if x_motor == scan.slow_motor_name:
        # Put y along the rows so the mosaic is drawn with x horizontal.
        n_slow, n_fast = grid_shape
        blocks = [blocks[i * n_fast + j] for j in range(n_fast) for i in range(n_slow)]
        grid_shape = (n_fast, n_slow)
        x_grid, y_grid, mon_grid = x_grid.T, y_grid.T, mon_grid.T

    print("Computing metrics and stitching mosaic...")
    pipeline_data = compute_pipeline_data(blocks, grid_shape)
    
    print("Saving HDF5 archive...")
    h5_path = args.output_dir / f"computed_metrics_scan{args.scan:04d}.h5"
    save_metrics_to_hdf5(pipeline_data, h5_path)
    
    print("Generating comprehensive visualization...")
    plot_filename = f"{args.h5_file.stem}_scan{args.scan:04d}_mosaic_X{args.roi_x}_Y{args.roi_y}_W{args.hws}.png"
    plot_path = args.output_dir / plot_filename
    
    plot_and_save_figure(
        data=pipeline_data, 
        x_grid=x_grid, 
        y_grid=y_grid, 
        mon_grid=mon_grid,
        motor_names=(x_motor, y_motor), 
        output_path=plot_path
    )


if __name__ == "__main__":
    main()