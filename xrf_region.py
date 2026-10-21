"""
Surface region of a map: which pixels belong to the flat surface being analysed.

A thin film covers the whole map, so every pixel is used (the default).  A
single crystal glued on a holder does not: the map shows a central flat facet,
neighbouring facets at other tilts (different incidence and take-off angles,
hence different intensities), the glue and the holder.  Only the facet whose
orientation is known (e.g. from the plane fitted to align ``sx``/``sy``) should
enter the average spectrum and the quantification.

A :class:`SurfaceRegion` is one or more polygons that are *included*, minus
optional polygons that are *excluded* (an inclusion, a dust grain, a crack
inside the facet).  A pixel is in the region when its **centre** is inside an
included polygon and inside no excluded one.

Vertices are given either

* in **motor** units: positions of the fast and slow motors of the mesh
  (``sx``, ``sy`` on BM32, in mm), read from the recorded positions of every
  point -- robust against a change of map size or step, and the units in
  which the region is reused on another map of the same facet; or
* in **pixel** units: (column, row) = (fast index, slow index) of the grid.

The region can be typed in (vertices known), or drawn on a map with
:class:`RegionSelector` in a Jupyter notebook (``%matplotlib widget``, ipympl),
and saved to / loaded from a small JSON file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.path import Path as MplPath

logger = logging.getLogger(__name__)

Polygon = np.ndarray          # (n_vertices, 2) array of (x, y) = (fast, slow)
_UNITS = ("motor", "pixel")


def _as_polygon(vertices) -> Polygon:
    """Vertices as an (n, 2) float array; at least three distinct vertices."""
    p = np.asarray(vertices, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"A polygon is a list of (x, y) vertices; got shape {p.shape}.")
    if len(p) > 1 and np.allclose(p[0], p[-1]):
        p = p[:-1]                                   # closing vertex repeated
    if len(np.unique(p, axis=0)) < 3:
        raise ValueError("A polygon needs at least three distinct vertices.")
    return p


def polygon_area(vertices) -> float:
    """Area of a simple polygon (shoelace formula), in the units of the vertices squared."""
    p = _as_polygon(vertices)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def points_in_polygon(x: np.ndarray, y: np.ndarray, vertices) -> np.ndarray:
    """Boolean array, True where the point (x, y) lies inside the polygon.

    Uses :meth:`matplotlib.path.Path.contains_points` (even-odd rule, the
    polygon is closed automatically).  Points exactly on an edge may fall on
    either side, which is irrelevant for pixel centres.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    pts = np.column_stack([x.ravel(), y.ravel()])
    inside = MplPath(_as_polygon(vertices)).contains_points(pts)
    return inside.reshape(x.shape)


def scan_coordinates(scan, units: str = "motor") -> Tuple[np.ndarray, np.ndarray]:
    """Per-pixel (x, y) = (fast, slow) coordinates of a mesh scan, flat arrays.

    Parameters
    ----------
    scan : BlissMeshScan (or any object with ``scan_shape``, ``fast_motor_name``,
        ``slow_motor_name`` and ``counters`` holding the motor positions)
    units : {"motor", "pixel"}
        ``"motor"``: recorded positions of the fast and slow motors (both must
        have been read, which ``BlissMeshScan.read_data`` does by default).
        ``"pixel"``: (column, row) indices on the ``scan_shape`` grid.
    """
    if units not in _UNITS:
        raise ValueError(f"units must be one of {_UNITS}, got {units!r}")
    n_slow, n_fast = scan.scan_shape
    if units == "pixel":
        row, col = np.indices((n_slow, n_fast))
        return col.ravel().astype(float), row.ravel().astype(float)
    try:
        x = np.asarray(scan.counters[scan.fast_motor_name], dtype=float).ravel()
        y = np.asarray(scan.counters[scan.slow_motor_name], dtype=float).ravel()
    except KeyError as exc:
        raise KeyError(f"motor position {exc} not loaded: read it with scan.read_data(), "
                       "or use units='pixel'.") from None
    return x, y


@dataclass
class SurfaceRegion:
    """Polygons defining the analysed surface of a map.

    Parameters
    ----------
    include : list of polygons
        Each polygon is a sequence of (x, y) vertices, x along the fast axis
        and y along the slow axis.  Pixels inside any of them are kept.  An
        empty list means the whole map.
    exclude : list of polygons
        Pixels inside any of these are removed (inclusions, dust, cracks).
    units : {"motor", "pixel"}
        Units of the vertices (see the module docstring).
    motors : (fast, slow) motor names, optional
        Recorded for motor units, and checked against the scan when the mask
        is built (a region drawn in ``sx``/``sy`` makes no sense on a map
        scanned with ``xech``/``yech``).
    name, note : str
        Free text, kept in the JSON file.

    Examples
    --------
    Whole map (thin film)::

        SurfaceRegion()

    Central facet of a crystal, vertices in mm (sx, sy), minus an inclusion::

        SurfaceRegion(include=[[(-3.84, -6.17), (-3.75, -6.17), (-3.74, -6.07), (-3.82, -6.06)]],
                      exclude=[[(-3.80, -6.10), (-3.79, -6.10), (-3.79, -6.09)]],
                      motors=("sx", "sy"), name="NIO295 top facet")
    """

    include: List[Polygon] = field(default_factory=list)
    exclude: List[Polygon] = field(default_factory=list)
    units: str = "motor"
    motors: Optional[Tuple[str, str]] = None
    name: str = ""
    note: str = ""

    def __post_init__(self):
        if self.units not in _UNITS:
            raise ValueError(f"units must be one of {_UNITS}, got {self.units!r}")
        self.include = [_as_polygon(p) for p in self.include]
        self.exclude = [_as_polygon(p) for p in self.exclude]
        if self.motors is not None:
            self.motors = tuple(self.motors)

    # ----------------------------------------------------------- constructors
    @classmethod
    def from_vertices(cls, vertices, units: str = "motor", exclude=(), **kw) -> "SurfaceRegion":
        """Region made of a single polygon (the usual case of one facet)."""
        return cls(include=[vertices], exclude=list(exclude), units=units, **kw)

    @classmethod
    def coerce(cls, region) -> "SurfaceRegion":
        """Accept what a notebook configuration naturally contains.

        ``None`` -> whole map; a ``SurfaceRegion`` -> itself; a ``str`` or
        ``Path`` -> :meth:`load`; a list of (x, y) vertices -> one polygon in
        motor units; a ``dict`` -> keyword arguments of the class.
        """
        if region is None:
            return cls()
        if isinstance(region, cls):
            return region
        if isinstance(region, (str, Path)):
            return cls.load(region)
        if isinstance(region, dict):
            return cls(**region)
        return cls.from_vertices(region)

    # ---------------------------------------------------------------- queries
    @property
    def is_full(self) -> bool:
        """True when no polygon is defined: the whole map is used."""
        return not self.include and not self.exclude

    def mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Boolean array, True for points inside the region (same shape as ``x``)."""
        x = np.asarray(x, dtype=float)
        keep = np.ones(x.shape, bool)
        if self.include:
            keep = np.zeros(x.shape, bool)
            for p in self.include:
                keep |= points_in_polygon(x, y, p)
        for p in self.exclude:
            keep &= ~points_in_polygon(x, y, p)
        return keep

    def scan_mask(self, scan) -> np.ndarray:
        """Per-pixel mask of a mesh scan, flat ``(n_pixels,)`` boolean array.

        Logs a warning when the region selects no pixel, or when it was defined
        with other motors than those of the scan.
        """
        if self.is_full:
            return np.ones(int(np.prod(scan.scan_shape)), bool)
        if self.units == "motor" and self.motors is not None:
            scan_motors = (scan.fast_motor_name, scan.slow_motor_name)
            if tuple(self.motors) != scan_motors:
                logger.warning("region defined in motors %s, scan uses %s",
                               self.motors, scan_motors)
        m = self.mask(*scan_coordinates(scan, self.units))
        if not m.any():
            logger.warning("the region %r contains no pixel of this scan", self.name)
        return m

    def area(self) -> float:
        """Included minus excluded polygon areas (overlaps not resolved), units^2."""
        return (sum(polygon_area(p) for p in self.include)
                - sum(polygon_area(p) for p in self.exclude))

    def __str__(self) -> str:
        if self.is_full:
            return "SurfaceRegion: whole map"
        u = "mm" if self.units == "motor" else "px"
        s = (f"SurfaceRegion {self.name!r}: {len(self.include)} included, "
             f"{len(self.exclude)} excluded polygon(s), {self.units} units")
        if self.include:
            s += f", area {self.area():.4g} {u}^2"
        return s

    # ------------------------------------------------------------- plotting
    def plot(self, ax, origin: Tuple[float, float] = (0.0, 0.0), scale: float = 1.0,
             include_kw: Optional[dict] = None, exclude_kw: Optional[dict] = None):
        """Draw the polygon outlines on ``ax``.

        Plotted coordinates are ``(vertex - origin) * scale``, so the outline
        can be laid over a map drawn in other units, e.g. µm from the map
        corner: ``origin=(x.min(), y.min()), scale=1000``.
        """
        from matplotlib.patches import Polygon as MplPolygon
        ikw = dict(fill=False, edgecolor="w", lw=1.4, ls="-")
        ikw.update(include_kw or {})
        ekw = dict(fill=False, edgecolor="r", lw=1.2, ls="--")
        ekw.update(exclude_kw or {})
        o = np.asarray(origin, dtype=float)
        patches = []
        for polys, kw in ((self.include, ikw), (self.exclude, ekw)):
            for p in polys:
                patches.append(ax.add_patch(MplPolygon((p - o) * scale, closed=True, **kw)))
        return patches

    # ----------------------------------------------------------------- I/O
    def to_dict(self) -> dict:
        return {"name": self.name, "note": self.note, "units": self.units,
                "motors": list(self.motors) if self.motors else None,
                "include": [p.tolist() for p in self.include],
                "exclude": [p.tolist() for p in self.exclude]}

    def save(self, path, **metadata) -> Path:
        """Write the region to JSON (extra ``metadata`` stored alongside, e.g. scan file)."""
        path = Path(path)
        d = self.to_dict()
        if metadata:
            d["metadata"] = {k: str(v) for k, v in metadata.items()}
        path.write_text(json.dumps(d, indent=2))
        return path

    @classmethod
    def load(cls, path) -> "SurfaceRegion":
        d = json.loads(Path(path).read_text())
        d.pop("metadata", None)
        return cls(**d)

    def to_code(self, precision: int = 5) -> str:
        """Python expression recreating the region, to paste into a notebook."""
        def fmt(polys):
            return "[" + ",\n             ".join(
                "[" + ", ".join(f"({x:.{precision}f}, {y:.{precision}f})" for x, y in p) + "]"
                for p in polys) + "]"
        args = [f"include={fmt(self.include)}"]
        if self.exclude:
            args.append(f"exclude={fmt(self.exclude)}")
        args.append(f"units={self.units!r}")
        if self.motors:
            args.append(f"motors={tuple(self.motors)!r}")
        if self.name:
            args.append(f"name={self.name!r}")
        return "SurfaceRegion(" + ",\n              ".join(args) + ")"


def map_image(corrected: np.ndarray, calibration=None,
              window_ev: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Per-pixel intensity to draw a region on: the sum of the normalised spectra.

    ``window_ev=(lo, hi)`` restricts the sum to an energy window (e.g. around
    a strong line, which shows the facets best); it needs ``calibration``.
    Non-finite pixels give NaN.
    """
    c = np.asarray(corrected, dtype=float)
    if window_ev is not None:
        if calibration is None:
            raise ValueError("window_ev needs a calibration.")
        lo, hi = (int(round(float(calibration.channel(e)))) for e in window_ev)
        c = c[:, max(lo, 0):max(hi, lo + 1)]
    with np.errstate(invalid="ignore"):
        return c.sum(axis=1)


class RegionSelector:
    """Draw a :class:`SurfaceRegion` on a map, in a Jupyter notebook.

    Needs an interactive Matplotlib backend: ``%matplotlib widget`` (ipympl)
    in Jupyter, or Qt/Tk in a script.  Usage::

        sel = RegionSelector(scan, image)      # shows the map
        # ... click the vertices of the facet, close the polygon on the first
        # vertex, adjust by dragging, then press "Add facet" (or key 'a').
        # Inclusions to remove: draw them and press "Add exclusion" (key 'x').
        region = sel.region                    # in the next cell
        region.save("processed/<scan>_region.json")

    Controls (buttons, or keys with the mouse over the map):

    * ``a``  add the polygon being drawn to the region (included)
    * ``x``  add it as an exclusion
    * ``u``  undo the last added polygon
    * ``Esc`` restart the polygon being drawn (Matplotlib's own binding)
    * drag a vertex to move it; ``shift``-drag moves the whole polygon

    Pixels outside the current region are greyed out, so what will be
    averaged is visible while drawing.

    Parameters
    ----------
    scan : BlissMeshScan
    image : (n_pixels,) or (n_slow, n_fast) array
        What to show, e.g. :func:`map_image`.
    units : {"motor", "pixel"}
        Units in which the vertices are recorded.
    region : SurfaceRegion, optional
        Starting region (e.g. a previous one, to refine).
    log : bool
        Logarithmic colour scale.
    """

    def __init__(self, scan, image: np.ndarray, units: str = "motor",
                 region: Optional[SurfaceRegion] = None, name: str = "",
                 log: bool = False, cmap: str = "viridis", figsize=(7.0, 6.2)):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from matplotlib.widgets import PolygonSelector

        self.scan = scan
        self.units = units
        self.shape = tuple(scan.scan_shape)
        self.x, self.y = scan_coordinates(scan, units)
        motors = (scan.fast_motor_name, scan.slow_motor_name) if units == "motor" else None
        region = SurfaceRegion.coerce(region)
        self._include = list(region.include)
        self._exclude = list(region.exclude)
        self._history: List[str] = []
        self._meta = dict(units=units, motors=motors, name=name or region.name,
                          note=region.note)

        backend = plt.get_backend().lower()
        if "inline" in backend or backend == "agg":
            logger.warning("Matplotlib backend %r is not interactive: run `%%matplotlib widget` "
                           "(ipympl) before creating the selector, or give the vertices.",
                           backend)

        img = np.asarray(image, dtype=float).reshape(self.shape)
        X, Y = self.x.reshape(self.shape), self.y.reshape(self.shape)
        with plt.ioff():
            self.fig, self.ax = plt.subplots(figsize=figsize)
        pos = img[np.isfinite(img) & (img > 0)]
        norm = LogNorm(*np.percentile(pos, [1, 99.5])) if log and pos.size else None
        kw = {} if norm else dict(vmin=np.nanpercentile(img, 1), vmax=np.nanpercentile(img, 99.5))
        self.ax.pcolormesh(X, Y, img, shading="nearest", cmap=cmap, norm=norm, **kw)
        self._shade = self.ax.pcolormesh(X, Y, np.zeros(self.shape), shading="nearest",
                                         cmap="Greys", vmin=0, vmax=1.5, alpha=0.0)
        if units == "motor":
            self.ax.set_aspect("equal")
            self.ax.set_xlabel(f"{scan.fast_motor_name} (mm)")
            self.ax.set_ylabel(f"{scan.slow_motor_name} (mm)")
        else:
            self.ax.set_xlabel("fast index (column)")
            self.ax.set_ylabel("slow index (row)")
        self._patches: list = []
        self._title()

        self.selector = PolygonSelector(self.ax, self._on_polygon, useblit=True,
                                        props=dict(color="w", lw=1.5),
                                        handle_props=dict(markersize=5, mfc="w", mec="k"))
        self._current: Optional[np.ndarray] = None
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._redraw()
        self._show()

    # -------------------------------------------------------------- display
    def _show(self):
        import matplotlib.pyplot as plt
        try:
            import ipywidgets as w
            from IPython.display import display
        except ImportError:                              # plain script: Qt/Tk window
            plt.show(block=False)
            return
        if not isinstance(self.fig.canvas, w.DOMWidget):  # inline or other backend
            display(self.fig)
            return
        b_add = w.Button(description="Add facet", button_style="success", icon="plus")
        b_exc = w.Button(description="Add exclusion", button_style="warning", icon="minus")
        b_undo = w.Button(description="Undo", icon="undo")
        b_clear = w.Button(description="Clear all", button_style="danger", icon="trash")
        b_add.on_click(lambda _: self.add(exclude=False))
        b_exc.on_click(lambda _: self.add(exclude=True))
        b_undo.on_click(lambda _: self.undo())
        b_clear.on_click(lambda _: self.clear())
        self._status = w.HTML()
        self._update_status()
        self.fig.canvas.header_visible = False
        display(w.VBox([w.HBox([b_add, b_exc, b_undo, b_clear]), self.fig.canvas, self._status]))

    def _title(self):
        self.ax.set_title("click vertices, close on the first one, then 'Add facet' (a) "
                          "or 'Add exclusion' (x)", fontsize=9, loc="left")

    def _update_status(self):
        if hasattr(self, "_status"):
            m = self.mask
            self._status.value = (f"<tt>{len(self._include)} facet polygon(s), "
                                  f"{len(self._exclude)} exclusion(s): "
                                  f"{int(m.sum())} / {m.size} pixels selected</tt>")

    def _redraw(self):
        for p in self._patches:
            p.remove()
        self._patches = self.region.plot(self.ax) if not self.region.is_full else []
        m = self.mask.reshape(self.shape)
        # grey veil on the excluded pixels only (selected ones are masked out)
        self._shade.set_array(np.ma.masked_array(np.ones(m.size), mask=m.ravel()))
        self._shade.set_alpha(0.0 if self.region.is_full else 0.6)
        self._update_status()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------- callbacks
    def _on_polygon(self, verts):
        self._current = np.asarray(verts, dtype=float)

    def _on_key(self, event):
        if event.key == "a":
            self.add(exclude=False)
        elif event.key == "x":
            self.add(exclude=True)
        elif event.key == "u":
            self.undo()

    # ---------------------------------------------------------------- actions
    def add(self, exclude: bool = False):
        """Add the polygon being drawn to the region (or to the exclusions)."""
        if self._current is None or len(self._current) < 3:
            logger.warning("no closed polygon to add: click the vertices and close the "
                           "polygon on its first vertex")
            return
        (self._exclude if exclude else self._include).append(_as_polygon(self._current))
        self._history.append("exclude" if exclude else "include")
        self._current = None
        self.selector.clear()
        self._redraw()

    def undo(self):
        """Remove the last added polygon."""
        if self._history:
            (self._exclude if self._history.pop() == "exclude" else self._include).pop()
            self._redraw()

    def clear(self):
        """Remove every polygon: back to the whole map."""
        self._include, self._exclude, self._history = [], [], []
        self._current = None
        self.selector.clear()
        self._redraw()

    # ---------------------------------------------------------------- result
    @property
    def region(self) -> SurfaceRegion:
        """The region drawn so far (the whole map if nothing was added)."""
        return SurfaceRegion(include=list(self._include), exclude=list(self._exclude),
                             **self._meta)

    @property
    def mask(self) -> np.ndarray:
        """Flat per-pixel mask of the region drawn so far."""
        return self.region.mask(self.x, self.y)
