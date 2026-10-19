"""
Module: bliss_hdf5_reader.py

Read any scan from a BLISS HDF5 master file (ESRF).

    from bliss_hdf5_reader import BlissScan, list_scans

    list_scans("sample_0001.h5")                 # what is in the file
    scan = BlissScan("sample_0001.h5", 12)       # any scan type
    print(scan.summary())
    mon  = scan.get("mon")                       # raw 1D counter (one value per point)
    img  = scan.get("mon", grid=True)            # same, shaped like the scan
    pos  = scan.positions("xech", grid=True)     # measured motor positions
    frm  = scan.read_frame(0)                    # first detector image

Scan geometry
-------------
``scan.shape`` follows the order of acquisition, slowest axis first:

    ct / sct / single point        ()
    ascan, a2scan, dscan, fscan,   (n,)
    loopscan, timescan, ...
    amesh / dmesh / fscan2d        (n_slow, n_fast)
    fscan3d                        (n_slowest, n_mid, n_fast)

The geometry is taken from the scan title when the command is recognised, and
otherwise inferred from the measured motor positions, so scans with a custom
title are still read correctly. ``scan.motors`` lists the scanned motors with
the dimension (``dim``) each one drives; several motors can share a dimension
(a2scan, a3scan, ...).

Aborted scans: ``scan.npoints`` is the number of points actually recorded and
``scan.npoints_planned`` the requested one. For 1D scans the shape is
``(npoints,)``; for meshes it stays the planned grid and missing points are
NaN in gridded data. Meshes acquired back-and-forth (snake) are detected from
the fast-motor positions and un-snaked in gridded data.

``BlissMeshScan`` is kept for existing scripts: a ``BlissScan`` that must be
two-dimensional and exposes the former ``fast_*`` / ``slow_*`` attributes.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# Datasets that are written as positioners but are time axes, not motors.
TIME_AXES = ("elapsed_time", "epoch", "epoch_trig", "timer_trig")
# Commands without a motor: one dimension indexed by time / point number.
TIME_SCANS = ("loopscan", "timescan", "ftimescan", "fscanloop", "limatake")
SINGLE_POINT_SCANS = ("ct", "sct")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _open(path: PathLike) -> h5py.File:
    """Open read-only without HDF5 file locking, so files that BLISS is still
    writing can be read at the beamline."""
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:  # h5py < 3.5 has no `locking` argument
        return h5py.File(path, "r")


def _to_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _to_str(value[()])
    return str(value)


def _read_scalar(group: h5py.Group, name: str) -> Any:
    if name not in group:
        return None
    value = group[name][()]
    if isinstance(value, (bytes, str)):
        return _to_str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _parse_key(scan: Union[int, str], subscan: int) -> Tuple[int, int, str]:
    """Accept 12, "12" or "12.1"."""
    if isinstance(scan, str) and "." in scan:
        number, sub = scan.split(".", 1)
        return int(number), int(sub), scan
    return int(scan), subscan, f"{int(scan)}.{subscan}"


def list_scans(h5_filepath: PathLike) -> List[Dict[str, Any]]:
    """
    One dict per scan entry in a master file, in scan order:
    key, number, title, npoints (planned, if recorded), end_reason, start_time.
    """
    rows = []
    with _open(h5_filepath) as f:
        for key in f:
            group = f[key]
            if not isinstance(group, h5py.Group):
                continue
            params = group.get("scan_parameters")
            rows.append({
                "key": key,
                "number": int(key.split(".")[0]) if key.split(".")[0].isdigit() else None,
                "title": _read_scalar(group, "title"),
                "npoints": _read_scalar(params, "npoints") if params is not None else None,
                "end_reason": _read_scalar(group, "end_reason"),
                "start_time": _read_scalar(group, "start_time"),
            })
    def order(row):
        parts = row["key"].split(".")
        return tuple(int(p) if p.isdigit() else 0 for p in parts)
    return sorted(rows, key=order)


@dataclass
class ScanMotor:
    """A motor driven by the scan. `start`/`stop` are as written in the title
    (relative to the start position for d-scans); `dim` is the scan dimension
    it moves along (0 = slowest)."""
    name: str
    dim: int
    npoints: int
    start: Optional[float] = None
    stop: Optional[float] = None
    relative: bool = False

    @property
    def step(self) -> Optional[float]:
        if self.start is None or self.stop is None or self.npoints < 2:
            return None
        return (self.stop - self.start) / (self.npoints - 1)


@dataclass
class FrameSource:
    """Where one detector frame lives on disk."""
    file: Path
    dataset: str
    index: int


# --------------------------------------------------------------------------- #
# title parsers: command -> (shape, motors, count_time) or None if not matched
# --------------------------------------------------------------------------- #
def _parse_step_scan(cmd: str, args: List[str]):
    """ascan/dscan/a2scan/d3scan/lineup:  (motor start stop)*N intervals time"""
    if len(args) < 5 or (len(args) - 2) % 3:
        return None
    n_motors = (len(args) - 2) // 3
    triples = [args[3 * i: 3 * i + 3] for i in range(n_motors)]
    if not all(_is_number(s) and _is_number(e) for _, s, e in triples):
        return None
    if not (args[-2].lstrip("-").isdigit() and _is_number(args[-1])):
        return None
    npts = int(args[-2]) + 1
    relative = cmd.startswith("d") or cmd == "lineup"
    motors = [ScanMotor(m, 0, npts, float(s), float(e), relative) for m, s, e in triples]
    return (npts,), motors, float(args[-1])


def _parse_mesh(cmd: str, args: List[str]):
    """amesh/dmesh: fast start stop intervals  slow start stop intervals  time.
    The FIRST motor is the fast one."""
    if len(args) < 9:
        return None
    try:
        fast = (args[0], float(args[1]), float(args[2]), int(args[3]) + 1)
        slow = (args[4], float(args[5]), float(args[6]), int(args[7]) + 1)
        count_time = float(args[8])
    except ValueError:
        return None
    relative = cmd == "dmesh"
    motors = [ScanMotor(slow[0], 0, slow[3], slow[1], slow[2], relative),
              ScanMotor(fast[0], 1, fast[3], fast[1], fast[2], relative)]
    return (slow[3], fast[3]), motors, count_time


def _parse_fscan_nd(cmd: str, args: List[str]):
    """fscan / fscan2d / fscan3d:  (motor start step npoints)*N acq_time [period].
    The LAST motor is the fast one (opposite of amesh)."""
    ndim = {"fscan": 1, "fscan2d": 2, "fscan3d": 3}[cmd]
    if len(args) < 4 * ndim + 1:
        return None
    motors, shape = [], []
    try:
        for d in range(ndim):
            name, start, step, npts = args[4 * d: 4 * d + 4]
            start, step, npts = float(start), float(step), int(npts)
            motors.append(ScanMotor(name, d, npts, start, start + step * (npts - 1)))
            shape.append(npts)
        count_time = float(args[4 * ndim])
    except ValueError:
        return None
    return tuple(shape), motors, count_time


def _parse_fnscan(cmd: str, args: List[str]):
    """f2scan/f3scan:  (motor start step)*N npoints acq_time [period]"""
    n_motors = int(cmd[1:-4])
    if len(args) < 3 * n_motors + 2:
        return None
    try:
        npts = int(args[3 * n_motors])
        motors = []
        for i in range(n_motors):
            name, start, step = args[3 * i: 3 * i + 3]
            start, step = float(start), float(step)
            motors.append(ScanMotor(name, 0, npts, start, start + step * (npts - 1)))
        count_time = float(args[3 * n_motors + 1])
    except ValueError:
        return None
    return (npts,), motors, count_time


def _parse_time_scan(cmd: str, args: List[str]):
    """loopscan npoints time | timescan time [npoints] | ftimescan time npoints [period]"""
    nums = [a for a in args if _is_number(a)]
    try:
        if cmd == "loopscan" and len(nums) >= 2:
            return (int(nums[0]),), [], float(nums[1])
        if cmd in ("timescan", "ftimescan") and nums:
            npts = int(float(nums[1])) if len(nums) > 1 else 0
            return (npts,), [], float(nums[0])   # npoints 0 = run until stopped
    except ValueError:
        return None
    return None


def _parse_single_point(cmd: str, args: List[str]):
    count_time = float(args[0]) if args and _is_number(args[0]) else None
    return (), [], count_time


def _parse_title(title: str):
    tokens = title.replace(",", " ").split()
    if not tokens:
        return "", None
    cmd, args = tokens[0], tokens[1:]
    if re.fullmatch(r"[ad]\d*scan|lineup", cmd):
        return cmd, _parse_step_scan(cmd, args)
    if cmd in ("amesh", "dmesh"):
        return cmd, _parse_mesh(cmd, args)
    if cmd in ("fscan", "fscan2d", "fscan3d"):
        return cmd, _parse_fscan_nd(cmd, args)
    if re.fullmatch(r"f\d+scan", cmd):
        return cmd, _parse_fnscan(cmd, args)
    if cmd in TIME_SCANS:
        return cmd, _parse_time_scan(cmd, args)
    if cmd in SINGLE_POINT_SCANS:
        return cmd, _parse_single_point(cmd, args)
    return cmd, None


# --------------------------------------------------------------------------- #
# main class
# --------------------------------------------------------------------------- #
class BlissScan:
    """
    One scan of a BLISS HDF5 master file, of any type.

    Instantiation only reads metadata. Data are read on demand with `get()`,
    `positions()` and `read_frame()`, or in bulk with `read_data()`.

    Args:
        h5_filepath: BLISS master file (<dataset>.h5).
        scan_number: scan number (12), or full entry key ("12.1").
        subscan: sub-scan index, used when `scan_number` is an integer.
        image_prefix: preferred image detector, matched against the start of
            the detector name (e.g. "eiger"); None = first image detector.
        image_suffix: file extension used when images have to be found by
            listing the scan folder (no virtual dataset in the master file).
        verbose: print the summary after `read_data()`.
    """

    def __init__(
        self,
        h5_filepath: PathLike,
        scan_number: Union[int, str],
        subscan: int = 1,
        image_prefix: Optional[str] = "eiger",
        image_suffix: str = ".h5",
        verbose: bool = False,
    ):
        self.h5_filepath = Path(h5_filepath)
        self.scan_number, self.subscan, self.scan_key = _parse_key(scan_number, subscan)
        self.image_prefix = image_prefix
        self.image_suffix = image_suffix
        self.verbose = verbose

        self.counters: Dict[str, np.ndarray] = {}   # raw data, as stored
        self.grids: Dict[str, np.ndarray] = {}      # scalar counters shaped like the scan
        self.image_path_list: List[Path] = []
        self.nb_images: int = 0                      # image files
        self.nb_frames: int = 0                      # detector frames
        self.scan_folder = self.h5_filepath.parent / f"scan{self.scan_number:04d}"

        self._read_metadata()

    # ------------------------------------------------------------ metadata
    def _read_metadata(self) -> None:
        if not self.h5_filepath.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_filepath}")

        with _open(self.h5_filepath) as f:
            if self.scan_key not in f:
                raise KeyError(f"Scan {self.scan_key} not found in {self.h5_filepath.name}")
            entry = f[self.scan_key]
            self.title: str = _read_scalar(entry, "title") or ""
            self.start_time: Optional[str] = _read_scalar(entry, "start_time")
            self.end_time: Optional[str] = _read_scalar(entry, "end_time")
            self.end_reason: Optional[str] = _read_scalar(entry, "end_reason")
            self.writer_status = (_read_scalar(entry["writer"], "status")
                                  if "writer" in entry else None)
            self.sample_name = (_read_scalar(entry["sample"], "name")
                                if "sample" in entry else None)

            params = entry.get("scan_parameters")
            npts_param = _read_scalar(params, "npoints") if params is not None else None
            time_param = _read_scalar(params, "count_time") if params is not None else None

            meas = entry["measurement"]
            self._datasets = {k: (d.shape, d.dtype) for k, d in meas.items()
                              if isinstance(d, h5py.Dataset)}
            self._positioner_names = self._find_positioners(entry, meas)
            self.npoints = self._count_acquired_points(npts_param)
            self._positions_cache = {
                name: np.asarray(meas[name][()], dtype=float)
                for name in self._positioner_names
                if self._datasets[name][0][:1] == (self.npoints,) and len(self._datasets[name][0]) == 1
            }

        self.command, parsed = _parse_title(self.title)
        self.scan_type = self.command          # former attribute name
        if parsed is not None:
            self.shape, self.motors, title_time = parsed
            self.geometry_source = "title"
        else:
            self.shape, self.motors = self._infer_geometry(npts_param)
            title_time = None
            self.geometry_source = "data"
            logger.info("Scan %s: title '%s' not recognised; geometry inferred from the "
                        "motor positions: shape %s", self.scan_key, self.title, self.shape)

        count_time = time_param if time_param is not None else title_time
        self.count_time: Optional[float] = float(count_time) if count_time is not None else None
        planned = int(np.prod(self.shape)) if self.shape else 1
        if self.shape and planned == 0:          # timescan without npoints
            planned = None
        if self.geometry_source == "data":
            planned = npts_param or self.npoints
        elif npts_param and planned and npts_param != planned:
            logger.warning("Scan %s: title gives %d points but scan_parameters %d; using the latter.",
                           self.scan_key, planned, npts_param)
            planned = npts_param
        self.npoints_planned: Optional[int] = planned

        # 0D/1D: the shape is what was actually recorded.
        if len(self.shape) <= 1:
            self.shape = () if (self.shape == () and self.npoints <= 1) else (self.npoints,)
        self.snake = self._detect_snake()

    @staticmethod
    def _find_positioners(entry: h5py.Group, meas: h5py.Group) -> List[str]:
        """Measurement datasets that are motors, time axes excluded: NXpositioner
        entries, plus encoder channels named after a motor (the fast motor of
        fscans is recorded as a detector channel)."""
        snapshot = entry.get("instrument/positioners_start")
        motor_names = set(snapshot.keys()) if isinstance(snapshot, h5py.Group) else set()
        names = []
        for name in meas:
            if name in motor_names and name not in TIME_AXES:
                names.append(name)
                continue
            link = meas.get(name, getlink=True)
            target = getattr(link, "path", "")
            if "/instrument/" not in target:
                continue
            inst_name = target.split("/instrument/", 1)[1].split("/", 1)[0]
            try:
                nx_class = _to_str(entry["instrument"][inst_name].attrs.get("NX_class", ""))
            except KeyError:
                continue
            if nx_class == "NXpositioner" and name not in TIME_AXES:
                names.append(name)
        return names

    def _count_acquired_points(self, npts_param: Optional[int]) -> int:
        """Points actually recorded: the most common first-dimension length of the
        measurement datasets, ignoring the 2n+1 '*_raw' channels of fast scans."""
        lengths = [shape[0] for shape in (s for s, _ in self._datasets.values()) if len(shape) >= 1]
        if not lengths:
            return 0
        if npts_param:
            lengths = [n for n in lengths if n <= npts_param] or lengths
        counts = Counter(lengths)
        top = max(counts.values())
        return max(n for n, c in counts.items() if c == top)

    def _infer_geometry(self, npts_planned: Optional[int] = None
                        ) -> Tuple[Tuple[int, ...], List[ScanMotor]]:
        """Geometry for titles that are not recognised: find the motors that move,
        and check whether they form a mesh (one motor stepping once per row)."""
        n = self.npoints
        if n <= 1:
            return (), []
        moving = {k: v for k, v in self._positions_cache.items()
                  if np.ptp(v) > 1e-9 * max(1.0, np.abs(v).max())}   # ignore rounding noise
        if not moving:
            return (n,), []

        mesh = self._infer_mesh(moving)
        if mesh is not None:
            (n_slow, n_fast), motors = mesh
            if npts_planned and npts_planned % n_fast == 0 and npts_planned // n_fast > n_slow:
                n_slow = npts_planned // n_fast          # aborted: keep the planned rows
                for m in motors:
                    if m.dim == 0:
                        m.npoints = n_slow
            return (n_slow, n_fast), motors

        # 1D: every motor that moves
        motors = [ScanMotor(k, 0, n, float(v[0]), float(v[-1])) for k, v in moving.items()]
        return (n,), motors

    @staticmethod
    def _row_length(pos: np.ndarray) -> Optional[int]:
        """Row length if `pos` is constant along rows and steps between rows
        (the slow motor of a mesh), else None."""
        n = pos.size
        d = np.abs(np.diff(pos))
        jumps = np.flatnonzero(d > 0.5 * d.max()) + 1
        row = int(jumps[0])
        # one jump per row boundary, evenly spaced, and at least two full rows
        if row < 2 or n < 2 * row or len(jumps) != (n - 1) // row \
                or np.any(np.diff(jumps) != row):
            return None
        full = pos[: (n // row) * row].reshape(-1, row)
        if np.ptp(full, axis=1).max() > 0.1 * np.median(d[jumps - 1]):
            return None                                  # moves along the row: not a slow motor
        return row

    @classmethod
    def _infer_mesh(cls, moving: Dict[str, np.ndarray]):
        """Return ((n_slow, n_fast), motors) if some motors change only every n_fast
        points while others repeat the same positions on every row, else None."""
        rows = {name: cls._row_length(pos) for name, pos in moving.items()}
        for row in sorted({r for r in rows.values() if r}):
            slow_names = [k for k, r in rows.items() if r == row]
            n = next(iter(moving.values())).size
            n_slow = -(-n // row)
            fast_candidates = []
            for fast_name, fast in moving.items():
                if fast_name in slow_names:
                    continue
                row0, row1 = fast[:row], fast[row:2 * row]
                span = np.ptp(row0)
                if span == 0:
                    continue
                # the fast motor repeats the same positions on every row (or reversed)
                mismatch = min(np.abs(row1 - row0).max(), np.abs(row1 - row0[::-1]).max())
                if mismatch < 0.1 * span:
                    fast_candidates.append(fast_name)
            if not fast_candidates:
                continue
            motors = [ScanMotor(name, 0, n_slow, float(moving[name][0]), float(moving[name][-1]))
                      for name in slow_names]
            motors += [ScanMotor(name, 1, row, float(moving[name][0]), float(moving[name][row - 1]))
                       for name in fast_candidates]
            return (n_slow, row), motors
        return None

    def _detect_snake(self) -> bool:
        """True if a 2D scan was acquired back-and-forth (odd rows reversed)."""
        if len(self.shape) != 2 or self.shape[0] < 2 or self.npoints < 2 * self.shape[1]:
            return False
        fast = [m for m in self.motors if m.dim == 1]
        if not fast or fast[0].name not in self._positions_cache:
            return False
        pos = self._positions_cache[fast[0].name]
        n_fast = self.shape[1]
        row0, row1 = pos[:n_fast], pos[n_fast:2 * n_fast]
        if np.ptp(row0) == 0:
            return False
        return bool(np.abs(row1 - row0[::-1]).max() < np.abs(row1 - row0).max())

    # ------------------------------------------------------------ properties
    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def is_complete(self) -> bool:
        if self.end_reason is not None:
            return self.end_reason == "SUCCESS"
        return self.npoints_planned is None or self.npoints >= self.npoints_planned

    @property
    def motor_names(self) -> List[str]:
        return [m.name for m in self.motors]

    @property
    def positioner_names(self) -> List[str]:
        """Every motor recorded at each point (scanned and dependent motors)."""
        return list(self._positioner_names)

    def motors_of_dim(self, dim: int) -> List[ScanMotor]:
        return [m for m in self.motors if m.dim == dim]

    @property
    def index_grid(self) -> np.ndarray:
        """Flat point index at each grid position (-1 where no point was recorded).
        Use it to place any per-point quantity on the scan grid."""
        planned = int(np.prod(self.shape)) if self.shape else 1
        idx = np.full(planned, -1, dtype=np.int64)
        n = min(self.npoints, planned)
        idx[:n] = np.arange(n)
        idx = idx.reshape(self.shape)
        if self.snake:
            idx[1::2] = idx[1::2, ::-1]
        return idx

    # ------------------------------------------------------------ data access
    def get_available_counters(self) -> Dict[str, Dict[str, Any]]:
        """Name -> {'shape', 'dtype'} for every dataset of the measurement group,
        without reading any data."""
        return {k: {"shape": s, "dtype": str(t)} for k, (s, t) in self._datasets.items()}

    list_counters = get_available_counters

    def get(self, name: str, grid: bool = False) -> np.ndarray:
        """
        Read one dataset of the measurement group.

        With grid=True the per-point axis is reshaped to `self.shape`
        (2D spectra become (n_slow, n_fast, n_channels), etc.).
        Counters already loaded by `read_data()` are not read again.
        """
        if name in self.counters:
            data = self.counters[name]
        else:
            if name not in self._datasets:
                raise KeyError(f"'{name}' not in scan {self.scan_key}. "
                               f"Available: {sorted(self._datasets)}")
            shape, _ = self._datasets[name]
            nbytes = int(np.prod(shape)) * self._datasets[name][1].itemsize
            if nbytes > 2 ** 31:
                logger.warning("Reading '%s' %s loads %.1f GB into memory; "
                               "use read_frame() for images.", name, shape, nbytes / 2 ** 30)
            with _open(self.h5_filepath) as f:
                data = f[f"{self.scan_key}/measurement/{name}"][()]
        return self.to_grid(data, name) if grid else data

    def __getitem__(self, name: str) -> np.ndarray:
        return self.get(name, grid=True)

    def positions(self, motor: str, grid: bool = False) -> np.ndarray:
        """Measured positions of a motor (motor units)."""
        return self.get(motor, grid=grid).astype(float)

    def to_grid(self, data: np.ndarray, name: str = "array") -> np.ndarray:
        """
        Place per-point data (first axis = point index) on the scan grid.

        Missing points of an aborted mesh are NaN (integer data are converted
        to float for that), back-and-forth rows are put back in order.
        Data whose first axis is not the number of points (e.g. the 2n+1
        '*_raw' channels of fast scans) cannot be gridded: ValueError.
        """
        data = np.asarray(data)
        if data.ndim == 0:
            return data
        n = data.shape[0]
        size = int(np.prod(self.shape)) if self.shape else 1
        limit = max(size, self.npoints_planned or 0)
        if n > limit:
            raise ValueError(f"'{name}' has {n} values for a scan of {limit} points; "
                             "it is not one value per point.")
        if n > size:            # aborted 1D scan: a counter one point ahead of the others
            data, n = data[:size], size
        if n == size and not self.snake:
            return data.reshape(self.shape + data.shape[1:])

        if n < size:
            logger.debug("'%s': %d of %d points recorded, missing points set to NaN.",
                         name, n, size)
        idx = self.index_grid
        out_dtype = data.dtype if n == size else np.result_type(data.dtype, np.float32)
        out = np.full(idx.shape + data.shape[1:], np.nan if n < size else 0, dtype=out_dtype)
        valid = (idx >= 0) & (idx < n)
        out[valid] = data[idx[valid]]
        return out

    def read_data(self, extra_counters: Optional[Sequence[str]] = None) -> None:
        """
        Load the scanned motors and the requested counters into `self.counters`
        (raw, as stored), put scalar counters on the scan grid in `self.grids`,
        and locate the detector image files.
        """
        names = [m.name for m in self.motors if m.name in self._datasets]
        for m in self.motors:
            if m.name not in self._datasets:
                logger.warning("Scanned motor '%s' not found in the measurement group.", m.name)
        missing = [c for c in (extra_counters or []) if c not in self._datasets]
        for c in missing:
            logger.warning("Counter '%s' not found in the measurement group.", c)
        names += [c for c in (extra_counters or []) if c in self._datasets and c not in names]

        with _open(self.h5_filepath) as f:
            meas = f[f"{self.scan_key}/measurement"]
            for name in names:
                if name not in self.counters:
                    self.counters[name] = meas[name][()]

        for name, data in self.counters.items():
            if data.ndim == 1:
                try:
                    self.grids[name] = self.to_grid(data, name)
                except ValueError as err:
                    logger.warning("Not gridded: %s", err)

        self._process_images()
        if self.verbose:
            print(self.summary())

    # ------------------------------------------------------------ images
    @property
    def image_detectors(self) -> List[str]:
        """Measurement datasets holding one 2D image per point."""
        return [k for k, (s, _) in self._datasets.items() if len(s) == 3]

    @property
    def spectrum_detectors(self) -> List[str]:
        """Measurement datasets holding one 1D spectrum (MCA, ...) per point."""
        return [k for k, (s, _) in self._datasets.items() if len(s) == 2]

    def _pick_detector(self, detector: Optional[str]) -> Optional[str]:
        if detector is not None:
            if detector not in self._datasets:
                raise KeyError(f"Detector '{detector}' not in scan {self.scan_key}.")
            return detector
        dets = self.image_detectors
        if self.image_prefix:
            preferred = [d for d in dets if d.startswith(self.image_prefix)]
            dets = preferred or dets
        return dets[0] if dets else None

    def frame_sources(self, detector: Optional[str] = None) -> List[FrameSource]:
        """
        File, internal dataset and index of every frame of an image detector,
        read from the virtual dataset of the master file. Empty list if the
        detector is not stored as a virtual dataset.
        """
        detector = self._pick_detector(detector)
        if detector is None:
            return []
        with _open(self.h5_filepath) as f:
            dset = f[f"{self.scan_key}/measurement/{detector}"]
            if not dset.is_virtual:
                return []
            sources = []
            for vs in dset.virtual_sources():
                (v0, *_), (v1, *_) = vs.vspace.get_select_bounds()
                (s0, *_), _ = vs.src_space.get_select_bounds()
                path = Path(vs.file_name)
                if not path.is_absolute():
                    path = self.h5_filepath.parent / path
                for k in range(v1 - v0 + 1):
                    sources.append(FrameSource(path, vs.dset_name, s0 + k))
        return sources

    def read_frame(self, index: int, detector: Optional[str] = None,
                   roi: Optional[Tuple[slice, slice]] = None) -> np.ndarray:
        """One detector frame by point index (through the master file).
        `roi` = (row slice, column slice) returns only that part of the frame."""
        detector = self._pick_detector(detector)
        if detector is None:
            raise KeyError(f"No image detector in scan {self.scan_key}.")
        with _open(self.h5_filepath) as f:
            dset = f[f"{self.scan_key}/measurement/{detector}"]
            return dset[(index,) + tuple(roi)] if roi is not None else dset[index]

    def iter_frames(self, detector: Optional[str] = None,
                    indices: Optional[Sequence[int]] = None,
                    roi: Optional[Tuple[slice, slice]] = None) -> Iterator[np.ndarray]:
        """Frames one at a time, keeping the file open (faster than repeated
        `read_frame` calls). `roi` as in `read_frame`."""
        detector = self._pick_detector(detector)
        if detector is None:
            return
        with _open(self.h5_filepath) as f:
            dset = f[f"{self.scan_key}/measurement/{detector}"]
            for i in (range(dset.shape[0]) if indices is None else indices):
                yield dset[(i,) + tuple(roi)] if roi is not None else dset[i]

    def _process_images(self) -> None:
        """Fill `image_path_list` (image files in acquisition order) and
        `nb_images`, `nb_frames`."""
        sources = self.frame_sources()
        if sources:
            self.image_path_list = list(dict.fromkeys(s.file for s in sources))
            self.nb_frames = len(sources)
        elif self.scan_folder.is_dir():
            # No virtual dataset: list the scan folder (top level only, so that
            # processed files saved in sub-folders are not picked up).
            pattern = f"{self.image_prefix or ''}*{self.image_suffix}"
            self.image_path_list = sorted(self.scan_folder.glob(pattern))
            self.nb_frames = len(self.image_path_list)
        else:
            self.image_path_list = []
            self.nb_frames = 0
            if self.image_detectors:
                logger.warning("Image folder not found: %s", self.scan_folder)
        self.nb_images = len(self.image_path_list)

    # ------------------------------------------------------------ misc
    def positioners_start(self) -> Dict[str, Any]:
        """Snapshot of every motor position at the start of the scan."""
        with _open(self.h5_filepath) as f:
            grp = f.get(f"{self.scan_key}/instrument/positioners_start")
            if grp is None:
                return {}
            out = {}
            for k, v in grp.items():
                if isinstance(v, h5py.Dataset):
                    value = v[()]
                    out[k] = value.item() if isinstance(value, np.generic) else value
            return out

    def get_relative_position_grid(self, motor_name: str, scale: float = 1000.0) -> np.ndarray:
        """Gridded positions relative to the first recorded point, times `scale`
        (default 1000: mm -> µm)."""
        grid = self.to_grid(self.positions(motor_name), motor_name)
        flat = grid.ravel()
        origin = flat[self.index_grid.ravel() == 0][0] if grid.ndim else grid
        return (grid - origin) * scale

    def summary(self) -> str:
        lines = [f"Scan {self.scan_key} of {self.h5_filepath.name}: {self.title}"]
        status = "complete" if self.is_complete else f"INCOMPLETE ({self.end_reason})"
        planned = f" of {self.npoints_planned}" if self.npoints_planned else ""
        lines.append(f"  shape {self.shape} [{self.geometry_source}], {self.npoints}{planned} "
                     f"points, {status}" + (", back-and-forth" if self.snake else ""))
        if self.count_time is not None:
            lines.append(f"  count time {self.count_time:g} s")
        for m in sorted(self.motors, key=lambda m: m.dim):
            rng = ""
            if m.start is not None and m.stop is not None:
                rel = " (relative)" if m.relative else ""
                step = f", step {m.step:.6g}" if m.step is not None else ""
                rng = f": {m.start:g} -> {m.stop:g}{rel}, {m.npoints} pts{step}"
            lines.append(f"  dim {m.dim}  {m.name}{rng}")
        if self.image_detectors:
            lines.append(f"  image detectors: {self.image_detectors}")
        if self.spectrum_detectors:
            lines.append(f"  spectrum detectors: {self.spectrum_detectors}")
        if self.nb_images:
            lines.append(f"  image files found: {self.nb_images}")
        if self.counters:
            lines.append(f"  loaded: {list(self.counters)}")
        return "\n".join(lines)

    def get_summary(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (f"<{type(self).__name__} {self.h5_filepath.name}:{self.scan_key} "
                f"'{self.title}' shape={self.shape}>")


class BlissMeshScan(BlissScan):
    """
    A two-dimensional scan (amesh, dmesh, fscan2d, or a mesh with a custom title),
    with the attributes of the former mesh-only reader:

        fast_motor_name, fast_start, fast_stop, fast_intervals, fast_steps,
        fast_step_size (same for slow_*), scan_shape = (slow, fast),
        expected_points, count_time, scan_type.

    Grids are (slow, fast): rows follow the slow motor, whatever its name.
    """

    def _read_metadata(self) -> None:
        super()._read_metadata()
        if self.ndim != 2:
            raise ValueError(f"Scan {self.scan_key} is not a 2D mesh scan "
                             f"(shape {self.shape}). Title: {self.title}. Use BlissScan.")
        for label, dim in (("slow", 0), ("fast", 1)):
            m = self.motors_of_dim(dim)[0]
            setattr(self, f"{label}_motor_name", m.name)
            setattr(self, f"{label}_start", m.start)
            setattr(self, f"{label}_stop", m.stop)
            setattr(self, f"{label}_steps", self.shape[dim])
            setattr(self, f"{label}_intervals", self.shape[dim] - 1)
            setattr(self, f"{label}_step_size", m.step or 0.0)
        self.scan_shape = self.shape
        self.expected_points = int(np.prod(self.shape))


# --------------------------------------------------------------------------- #
def main() -> None:
    """Command line: list the scans of a file, or summarise some of them.

        python bliss_hdf5_reader.py file.h5            # list scans
        python bliss_hdf5_reader.py file.h5 3 7 12.2   # summary of scans 3, 7, 12.2
    """
    import argparse
    parser = argparse.ArgumentParser(description=main.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5_file", type=Path)
    parser.add_argument("scans", nargs="*")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.scans:
        for row in list_scans(args.h5_file):
            npts = row["npoints"] if row["npoints"] is not None else "-"
            print(f"{row['key']:>7}  {str(npts):>7}  {str(row['end_reason'] or ''):<11} {row['title']}")
        return
    for key in args.scans:
        scan = BlissScan(args.h5_file, key)
        print(scan.summary())
        counters = scan.get_available_counters()
        print(f"  {len(counters)} datasets: " + ", ".join(
            f"{k}{list(v['shape'][1:]) if len(v['shape']) > 1 else ''}" for k, v in counters.items()))
        print()


if __name__ == "__main__":
    main()
