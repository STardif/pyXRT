"""
Module: xrf_matrix.py

Matrix effects in X-ray fluorescence under polychromatic (white-beam)
excitation: primary absorption and secondary fluorescence in a homogeneous
layer of arbitrary thickness, from the thin-film limit to a bulk sample.

Geometry and notation
---------------------
A homogeneous layer of density ``rho`` and thickness ``t`` (mass thickness
``M = rho t``) lies on an inert substrate.  The incident beam meets the surface
at ``psi_in`` and the detector sees it at the take-off angle ``psi_out`` (both
measured from the surface).  ``m`` is mass depth (g/cm^2), ``mu(E)`` the total
mass attenuation coefficient of the layer (cm^2/g), ``Q_il(E)`` the partial
fluorescence cross-section of line ``l`` of element ``i``
(``xraylib.CS_FluorLine_Kissel``, cm^2/g, including subshell ionisation,
Coster-Kronig transfer and fluorescence yield) and ``dPhi/dE`` the incident
spectral density per keV.

Primary fluorescence, per unit mass fraction of ``i``::

    P_il = int dE  dPhi/dE / sin(psi_in) * Q_il(E) * F(a + b)

    a = mu(E) / sin(psi_in),   b = mu(E_l) / sin(psi_out),
    F(c) = (1 - exp(-c M)) / c          (-> M for a thin film, 1/c for bulk)

Secondary fluorescence: line ``j`` of element ``j`` (energy ``E_j``, produced
by the primary beam at depth ``m``) is emitted isotropically and absorbed at
depth ``m'`` by element ``i``, which then emits line ``l``.  Writing the
isotropic point-source kernel as ``(1/2) int_0^1 exp(-mu_j |m - m'| / u) du/u``
(``u = cos theta``), the depth integrals have a closed form for every ``u``::

    S_il = sum_j  (1/2) Q_il(E_j)  w_j
           int dE dPhi/dE / sin(psi_in) * Q_j(E) * int_0^1 D(a, b, mu_j/u; M) du/u

    D(a, b, k; M) = int_0^M int_0^M exp(-a m - b m' - k |m - m'|) dm dm'
                  = [F(a+k) - F(a+b)] / (b - k) + [F(b+k) - F(a+b)] / (a - k)

For ``M -> inf`` this reduces to the classical bulk result
``[ln(1 + a/mu_j)/a + ln(1 + b/mu_j)/b] / (2 (a + b))``; for ``M -> 0`` it
reproduces the logarithmic thin-film enhancement.  The ``u`` integral is done
on a logarithmic grid, which resolves the grazing paths that dominate in thin
films.

Both terms share the solid-angle factor and the detection efficiency, which
are applied elsewhere (``xrf_fit.Detector.efficiency``), so

    measured line intensity  ∝  w_i * (P_il + S_il) * eff(E_l).

Not included: tertiary fluorescence, fluorescence excited in the substrate,
excitation by scattered radiation, multilayers, surface roughness.  The
substrate is assumed not to emit lines able to excite the layer (true for Si
under the layers studied here: Si K at 1.74 keV lies below every edge of
interest).

Authors: Samuel Tardif, with Claude (Anthropic).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import xraylib
from scipy.integrate import trapezoid

from xrf_beamline import per_unit_energy

logger = logging.getLogger(__name__)
xraylib.SetErrorMessages(0)

__all__ = [
    "Geometry",
    "Layer",
    "MatrixModel",
    "quantify_layer",
]


# =============================================================================
# Cached xraylib look-ups on an energy grid
# =============================================================================

_CACHE: Dict[tuple, np.ndarray] = {}


def _grid_key(e: np.ndarray) -> tuple:
    return (e.size, hash(e.tobytes()))


def _cs_total(z: int, e_kev: np.ndarray) -> np.ndarray:
    key = ("tot", z, _grid_key(e_kev))
    if key not in _CACHE:
        _CACHE[key] = np.array([xraylib.CS_Total(z, float(e)) for e in e_kev])
    return _CACHE[key]


def _cs_fluor(z: int, macro: int, e_kev: np.ndarray) -> np.ndarray:
    key = ("fl", z, macro, _grid_key(e_kev))
    if key not in _CACHE:
        out = np.zeros(e_kev.size)
        for n, e in enumerate(e_kev):
            try:
                out[n] = max(xraylib.CS_FluorLine_Kissel(z, macro, float(e)), 0.0)
            except ValueError:
                pass
        _CACHE[key] = out
    return _CACHE[key]


def _cs_fluor_scalar(z: int, macro: int, e_kev: float) -> float:
    try:
        return max(xraylib.CS_FluorLine_Kissel(z, macro, float(e_kev)), 0.0)
    except ValueError:
        return 0.0


#: IUPAC line macros only (see xrf_fit for why Siegbahn names are excluded).
_IUPAC_RE = re.compile(r"^(K|L[1-3]|M[1-5])(K|L[1-3]|M[1-5]|N[1-7]|O[1-7]|P[1-5])_LINE$")
_IUPAC_LINES = {n: getattr(xraylib, n) for n in dir(xraylib) if _IUPAC_RE.match(n)}


# =============================================================================
# Geometry and sample
# =============================================================================

@dataclass(frozen=True)
class Geometry:
    """Incidence and take-off angles, in degrees from the sample **surface**."""

    incidence_deg: float = 40.0
    takeoff_deg: float = 40.0

    @property
    def sin_in(self) -> float:
        return float(np.sin(np.radians(self.incidence_deg)))

    @property
    def sin_out(self) -> float:
        return float(np.sin(np.radians(self.takeoff_deg)))


class Layer:
    """A homogeneous layer.

    Parameters
    ----------
    composition : str or mapping
        A chemical formula (``"In0.5Ga0.5As"``) or ``{symbol: atoms}``
        (``{"In": 0.5, "Ga": 0.5, "As": 1}``); only atomic *proportions* matter.
    density_g_cm3 : float
    thickness_um : float or None
        ``None`` (or ``inf``) for a bulk sample.
    """

    def __init__(self, composition: Union[str, Mapping[str, float]],
                 density_g_cm3: float, thickness_um: Optional[float] = None):
        if isinstance(composition, str):
            formula = composition
        else:
            formula = "".join(f"{el}{n:.6g}" for el, n in composition.items() if n > 0)
        parsed = xraylib.CompoundParser(formula)
        self.formula = formula
        self.mass_fractions: Dict[int, float] = dict(zip(parsed["Elements"],
                                                         parsed["massFractions"]))
        atoms = dict(zip(parsed["Elements"], parsed["nAtoms"]))
        self.atoms: Dict[str, float] = {xraylib.AtomicNumberToSymbol(z): float(n)
                                        for z, n in atoms.items()}
        self.density_g_cm3 = float(density_g_cm3)
        self.thickness_um = (None if thickness_um is None or not np.isfinite(thickness_um)
                             else float(thickness_um))

    @property
    def mass_thickness(self) -> float:
        """``rho t`` in g/cm^2 (``inf`` for bulk)."""
        if self.thickness_um is None:
            return np.inf
        return self.density_g_cm3 * self.thickness_um * 1e-4

    def mu(self, e_kev: np.ndarray | float) -> np.ndarray | float:
        """Total mass attenuation coefficient, cm^2/g."""
        scalar = np.ndim(e_kev) == 0
        e = np.atleast_1d(np.asarray(e_kev, dtype=float))
        out = np.zeros(e.size)
        for z, w in self.mass_fractions.items():
            out += w * (_cs_total(z, e) if e.size > 8 else
                        np.array([xraylib.CS_Total(z, float(x)) for x in e]))
        return float(out[0]) if scalar else out

    def with_composition(self, composition: Union[str, Mapping[str, float]]) -> "Layer":
        return Layer(composition, self.density_g_cm3, self.thickness_um)

    def __repr__(self) -> str:
        t = "bulk" if self.thickness_um is None else f"{self.thickness_um:g} um"
        return f"Layer({self.formula!r}, rho={self.density_g_cm3:g} g/cm3, {t})"


# =============================================================================
# Depth integrals
# =============================================================================

def _F(c: np.ndarray, M: float) -> np.ndarray:
    """``int_0^M exp(-c m) dm``."""
    if not np.isfinite(M):
        return 1.0 / c
    return -np.expm1(-c * M) / c


def _dF(c: np.ndarray, M: float) -> np.ndarray:
    """``d F / d c``."""
    if not np.isfinite(M):
        return -1.0 / c ** 2
    e = np.exp(-c * M)
    return (c * M * e - (1.0 - e)) / c ** 2


def _D(a, b, k, M: float, rtol: float = 1e-4):
    """``int_0^M int_0^M exp(-a m - b m' - k|m - m'|) dm dm'`` (closed form)."""
    a, b, k = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float), np.asarray(k, float))
    f_ab = _F(a + b, M)
    with np.errstate(divide="ignore", invalid="ignore"):
        near_b = np.abs(b - k) <= rtol * (b + k)
        t1 = np.where(near_b, -_dF(a + k, M), (_F(a + k, M) - f_ab) / (b - k))
        near_a = np.abs(a - k) <= rtol * (a + k)
        t2 = np.where(near_a, -_dF(b + k, M), (_F(b + k, M) - f_ab) / (a - k))
    return t1 + t2


# =============================================================================
# The model
# =============================================================================

@dataclass
class _Exciter:
    symbol: str
    name: str
    energy_kev: float
    mass_fraction: float
    cs: np.ndarray          # Q_j(E) on the SF grid
    mu: float               # layer attenuation at E_j


class MatrixModel:
    """Primary and secondary fluorescence sensitivities for one layer.

    Parameters
    ----------
    layer : Layer
    geometry : Geometry
    energy_kev, flux_01bw : (n,) arrays
        Incident spectrum, **per 0.1 % bandwidth** as returned by
        ``xrf_beamline.Beamline.spectrum``; converted to per keV internally.
        A uniform grid with ~10 eV steps is appropriate.
    secondary : bool
        Include secondary fluorescence.
    sf_step_kev : float
        The secondary term is smooth in E and is evaluated on a grid decimated
        to about this step (default 40 eV).
    n_u : int
        Points of the logarithmic direction-cosine grid for the SF kernel.
    exciter_min_rel : float
        Exciting lines weaker than this fraction of their element's strongest
        line (primary production in the layer) are ignored.
    """

    def __init__(self, layer: Layer, geometry: Geometry,
                 energy_kev: np.ndarray, flux_01bw: np.ndarray,
                 secondary: bool = True, sf_step_kev: float = 0.04,
                 n_u: int = 160, exciter_min_rel: float = 1e-3):
        self.layer = layer
        self.geometry = geometry
        self.energy = np.asarray(energy_kev, dtype=float)
        self.flux_01bw = np.asarray(flux_01bw, dtype=float)
        self.density = per_unit_energy(self.energy, self.flux_01bw)
        self.secondary_enabled = bool(secondary)
        self.sf_step_kev = sf_step_kev
        self.n_u = int(n_u)
        self.exciter_min_rel = exciter_min_rel

        self._M = layer.mass_thickness
        self._mu_E = layer.mu(self.energy)
        step = float(np.median(np.diff(self.energy)))
        self._dec = max(1, int(round(sf_step_kev / step)))
        self._E_sf = self.energy[::self._dec]
        self._phi_sf = self.density[::self._dec]
        self._mu_sf = self._mu_E[::self._dec]
        s = np.linspace(np.log(1e-7), 0.0, self.n_u)
        self._s, self._u = s, np.exp(s)
        self._exciters: Optional[List[_Exciter]] = None

    # ------------------------------------------------------------------ setup

    def with_layer(self, layer: Layer) -> "MatrixModel":
        """Same beam, geometry and numerics, different layer."""
        return MatrixModel(layer, self.geometry, self.energy, self.flux_01bw,
                           secondary=self.secondary_enabled, sf_step_kev=self.sf_step_kev,
                           n_u=self.n_u, exciter_min_rel=self.exciter_min_rel)

    @property
    def exciters(self) -> List[_Exciter]:
        """Lines of the layer's own elements that can drive secondary fluorescence."""
        if self._exciters is None:
            ex: List[_Exciter] = []
            sin_in = self.geometry.sin_in
            for z, w in self.layer.mass_fractions.items():
                sym = xraylib.AtomicNumberToSymbol(z)
                cands = []
                for name, macro in _IUPAC_LINES.items():
                    try:
                        e_j = xraylib.LineEnergy(z, macro)
                    except ValueError:
                        continue
                    if e_j < 1.0:
                        continue
                    cs = _cs_fluor(z, macro, self._E_sf)
                    if not cs.any():
                        continue
                    strength = trapezoid(self._phi_sf / sin_in * cs, self._E_sf)
                    cands.append((name[:-5], e_j, cs, strength))
                if not cands:
                    continue
                smax = max(c[3] for c in cands)
                for name, e_j, cs, st in cands:
                    if st >= self.exciter_min_rel * smax:
                        ex.append(_Exciter(sym, name, e_j, w, cs, self.layer.mu(e_j)))
            self._exciters = ex
            logger.info("%d exciting lines in %s", len(ex), self.layer.formula)
        return self._exciters

    # ------------------------------------------------------------ physics

    def primary(self, z: int, macro: int, line_energy_kev: float) -> float:
        """``P_il`` — primary sensitivity per unit mass fraction."""
        g = self.geometry
        q = _cs_fluor(z, macro, self.energy)
        if not q.any():
            return 0.0
        a = self._mu_E / g.sin_in
        b = self.layer.mu(line_energy_kev) / g.sin_out
        return float(trapezoid(self.density / g.sin_in * q * _F(a + b, self._M), self.energy))

    def secondary(self, z: int, macro: int, line_energy_kev: float,
                  by_exciter: bool = False):
        """``S_il`` — secondary sensitivity per unit mass fraction.

        With ``by_exciter=True`` also returns ``{exciter: contribution}``.
        """
        if not self.secondary_enabled:
            return (0.0, {}) if by_exciter else 0.0
        g = self.geometry
        a = (self._mu_sf / g.sin_in)[:, None]
        b = self.layer.mu(line_energy_kev) / g.sin_out
        total, parts = 0.0, {}
        for ex in self.exciters:
            q_ij = _cs_fluor_scalar(z, macro, ex.energy_kev)
            if q_ij <= 0.0:
                continue
            k = (ex.mu / self._u)[None, :]
            kernel = trapezoid(_D(a, b, k, self._M), self._s, axis=1)      # int D du/u
            val = 0.5 * q_ij * ex.mass_fraction * float(
                trapezoid(self._phi_sf / g.sin_in * ex.cs * kernel, self._E_sf))
            total += val
            parts[f"{ex.symbol} {ex.name}"] = val
        return (total, parts) if by_exciter else total

    def sensitivity(self, z: int, macro: int, line_energy_kev: float) -> float:
        """``P_il + S_il``."""
        p = self.primary(z, macro, line_energy_kev)
        if p <= 0.0:
            return 0.0
        return p + self.secondary(z, macro, line_energy_kev)

    def enhancement(self, z: int, macro: int, line_energy_kev: float) -> float:
        """Secondary / primary."""
        p = self.primary(z, macro, line_energy_kev)
        return self.secondary(z, macro, line_energy_kev) / p if p > 0 else 0.0

    def absorption_factor(self, z: int, macro: int, line_energy_kev: float) -> float:
        """Primary intensity relative to an absorption-free layer of the same
        mass thickness (thin-film limit); 1 means no self-absorption.
        Undefined (returns NaN) for a bulk layer."""
        if not np.isfinite(self._M):
            return float("nan")
        g = self.geometry
        q = _cs_fluor(z, macro, self.energy)
        thin = trapezoid(self.density / g.sin_in * q * self._M, self.energy)
        return self.primary(z, macro, line_energy_kev) / thin if thin > 0 else float("nan")


# =============================================================================
# Quantification by iteration on the composition
# =============================================================================

def _line(group):
    """(Z, macro, energy keV) of a group's reference line."""
    z = xraylib.SymbolToAtomicNumber(group.element)
    name = group.reference.name
    if "+" in name:
        raise ValueError(f"{group.label}: merged reference line {name!r} is not supported "
                         "for matrix quantification; build the groups with merge_within_ev=0.")
    return z, getattr(xraylib, name + "_LINE"), group.reference.energy_ev / 1000.0


def quantify_layer(areas: Mapping[str, float], groups: Mapping[str, object],
                   matrix: MatrixModel, detector, reference: str,
                   analytes: Sequence[str], fixed_atoms: Optional[Mapping[str, float]] = None,
                   max_iter: int = 50, tol: float = 1e-6) -> dict:
    """Self-consistent composition of a layer from fitted reference-line areas.

    Each analyte is expressed as atoms per atom of the ``reference`` group's
    element::

        n_i / n_ref = (A_i / A_ref) * (S_ref W_ref) / (S_i W_i)

    with ``S = (P + S_sec) * eff`` evaluated for the *current* composition.
    The layer composition is then rebuilt from these ratios (plus
    ``fixed_atoms``, for unmeasured elements such as O, given per reference
    atom) and the procedure repeated until the ratios change by less than
    ``tol``.  Density and thickness are held fixed.

    Parameters
    ----------
    areas : {label: reference-line area}
        E.g. ``FitResult.areas``.
    groups : {label: LineGroup}
    matrix : MatrixModel
        Its layer supplies the starting composition, density and thickness.
    detector : xrf_fit.Detector
        For the detection efficiency of each reference line.
    reference : str
        Label of the reference group, e.g. ``"As K"``.
    analytes : sequence of str
        Labels of the groups to quantify, e.g. ``["Ga K", "In L"]``.
    fixed_atoms : {symbol: atoms per reference atom}, optional

    Returns
    -------
    dict with keys ``ratios`` ({symbol: n/n_ref}), ``layer`` (converged
    Layer), ``matrix`` (MatrixModel for it), ``k`` ({label: sensitivity
    factor}), ``enhancement`` ({label: S_sec/P}), ``absorption`` ({label:
    primary absorption factor}), ``iterations`` and ``history``.
    """
    fixed_atoms = dict(fixed_atoms or {})
    g_ref = groups[reference]
    ref_el = g_ref.element
    w_ref = xraylib.AtomicWeight(xraylib.SymbolToAtomicNumber(ref_el))

    def sens(mm, g):
        z, macro, e = _line(g)
        return mm.sensitivity(z, macro, e) * detector.efficiency(e * 1000.0)

    atoms0 = matrix.layer.atoms
    n0 = atoms0.get(ref_el, 1.0)
    ratios = {el: n / n0 for el, n in atoms0.items()}
    history = [dict(ratios)]
    mm = matrix
    k = {}
    it = 0
    for it in range(1, max_iter + 1):
        s_ref = sens(mm, g_ref)
        new = {ref_el: 1.0}
        new.update(fixed_atoms)
        for lab in analytes:
            g = groups[lab]
            w_i = xraylib.AtomicWeight(xraylib.SymbolToAtomicNumber(g.element))
            k[lab] = (s_ref * w_ref) / (sens(mm, g) * w_i)
            new[g.element] = float(areas[lab] / areas[reference] * k[lab])
        change = max(abs(new.get(el, 0.0) - ratios.get(el, 0.0)) for el in set(new) | set(ratios))
        ratios = new
        history.append(dict(ratios))
        mm = mm.with_layer(mm.layer.with_composition(ratios))
        if change < tol:
            break
    else:
        logger.warning("quantify_layer did not converge in %d iterations", max_iter)

    enh, absf = {}, {}
    for lab in [reference, *analytes]:
        z, macro, e = _line(groups[lab])
        enh[lab] = mm.enhancement(z, macro, e)
        absf[lab] = mm.absorption_factor(z, macro, e)
    return dict(ratios=ratios, layer=mm.layer, matrix=mm, k=k, enhancement=enh,
                absorption=absf, iterations=it, history=history)
