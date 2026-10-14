"""
Module: xrf_matrix.py

Matrix effects in X-ray fluorescence under polychromatic (white-beam)
excitation: primary absorption and secondary fluorescence in a stack of
homogeneous layers (e.g. cap / film / substrate), each of arbitrary thickness,
the last one possibly bulk.

Geometry and notation
---------------------
Layers are numbered from the surface, ``k = 0, 1, ...``; layer ``k`` has
density ``rho_k``, thickness ``t_k`` and mass thickness ``M_k = rho_k t_k``
(``inf`` for a bulk last layer).  The incident beam meets the surface at
``psi_in`` and the detector sees it at the take-off angle ``psi_out`` (both from
the surface).  ``m`` is mass depth *within* a layer, ``mu_k(E)`` the total mass
attenuation coefficient of layer ``k``, ``Q_il(E)`` the partial fluorescence
cross-section of line ``l`` of element ``i`` (``xraylib.CS_FluorLine_Kissel``,
cm^2/g, including subshell ionisation, Coster-Kronig transfer and fluorescence
yield) and ``dPhi/dE`` the incident spectral density per keV.  Write

    a_k(E) = mu_k(E) / sin(psi_in),      b_k = mu_k(E_l) / sin(psi_out),
    F(c; M) = (1 - exp(-c M)) / c,       G(x, y; M) = (exp(-y M) - exp(-x M)) / (x - y)

Primary fluorescence of line ``l`` from layer ``k``, per unit mass fraction of
``i`` in that layer::

    P_il^k = T_out^k * int dE dPhi/dE / sin(psi_in) * Q_il(E) * T_in^k(E) * F(a_k + b_k; M_k)

with the overlayer transmissions ``T_in^k = exp(-sum_{c<k} a_c M_c)`` and
``T_out^k = exp(-sum_{c<k} b_c M_c)``.

Secondary fluorescence: line ``j`` (energy ``E_j``) produced by the beam in
layer ``a`` is emitted isotropically and absorbed in layer ``k`` by element
``i``, which emits line ``l``.  With the isotropic point-source kernel
``(1/2) int_0^1 exp(-mu x / u) du/u`` (``u = cos theta``, ``k_c = mu_c(E_j)/u``)
the depth integrals are closed-form for every ``u``::

    S_il^k = T_out^k * sum_{a, j} (1/2) Q_il(E_j) w_j^a
             int dE dPhi/dE / sin(psi_in) * Q_j(E) * T_in^a(E) * int_0^1 K_ak du/u

    a == k :  K = D(a_k, b_k, k_k; M_k)
              = [F(a_k+k_k) - F(a_k+b_k)] / (b_k - k_k) + [F(b_k+k_k) - F(a_k+b_k)] / (a_k - k_k)
    a <  k :  K = G(a_a, k_a; M_a) * exp(-sum_{a<c<k} k_c M_c) * F(b_k + k_k; M_k)
    a >  k :  K = F(a_a + k_a; M_a) * exp(-sum_{k<c<a} k_c M_c) * G(b_k, k_k; M_k)

For a single bulk layer this reduces to the classical
``[ln(1 + a/mu_j)/a + ln(1 + b/mu_j)/b] / (2 (a + b))``; for a thin layer it
gives the logarithmic thin-film enhancement.  The ``u`` integral is done on a
logarithmic grid, which resolves the grazing paths that dominate in thin films.

Both terms share the solid-angle factor and the detection efficiency (applied
in ``xrf_fit.Detector.efficiency``), so the measured intensity of line ``l`` is

    I_il  ∝  eff(E_l) * sum_k w_i^k (P_il^k + S_il^k).

Not included: tertiary fluorescence, excitation by scattered radiation,
interface roughness or interdiffusion, lateral inhomogeneity.

Authors: Samuel Tardif, with Claude (Anthropic).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Union

import numpy as np
import xraylib
from scipy.integrate import trapezoid

from xrf_beamline import per_unit_energy

logger = logging.getLogger(__name__)
xraylib.SetErrorMessages(0)

__all__ = [
    "Geometry",
    "Layer",
    "Stack",
    "MatrixModel",
    "quantify_layer",
    "atom_ratio",
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
        ``None`` (or ``inf``) for a bulk sample; only the last layer of a
        :class:`Stack` may be bulk.
    name : str
        Optional label ("cap", "film", "substrate").
    """

    def __init__(self, composition: Union[str, Mapping[str, float]],
                 density_g_cm3: float, thickness_um: Optional[float] = None,
                 name: str = ""):
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
        self.name = name

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
        return Layer(composition, self.density_g_cm3, self.thickness_um, self.name)

    def with_thickness(self, thickness_um: Optional[float]) -> "Layer":
        return Layer(self.atoms, self.density_g_cm3, thickness_um, self.name)

    def __repr__(self) -> str:
        t = "bulk" if self.thickness_um is None else f"{self.thickness_um:g} um"
        nm = f"{self.name}: " if self.name else ""
        return f"Layer({nm}{self.formula!r}, rho={self.density_g_cm3:g} g/cm3, {t})"


class Stack:
    """Ordered layers, **surface first**; only the last may be bulk."""

    def __init__(self, layers: Sequence[Layer]):
        layers = list(layers)
        if not layers:
            raise ValueError("A stack needs at least one layer.")
        for lay in layers[:-1]:
            if lay.thickness_um is None:
                raise ValueError(f"{lay}: only the last layer of a stack may be bulk.")
        self.layers = layers

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, k: int) -> Layer:
        return self.layers[k]

    def __iter__(self) -> Iterator[Layer]:
        return iter(self.layers)

    def replace(self, index: int, layer: Layer) -> "Stack":
        new = list(self.layers)
        new[index] = layer
        return Stack(new)

    def __repr__(self) -> str:
        return "Stack(" + " / ".join(repr(l) for l in self.layers) + ")"


# =============================================================================
# Depth integrals
# =============================================================================

def _F(c: np.ndarray, M: float) -> np.ndarray:
    """``int_0^M exp(-c m) dm``  (c > 0)."""
    if not np.isfinite(M):
        return 1.0 / c
    return -np.expm1(-c * M) / c


def _dF(c: np.ndarray, M: float) -> np.ndarray:
    """``d F / d c``."""
    if not np.isfinite(M):
        return -1.0 / c ** 2
    e = np.exp(-c * M)
    return (c * M * e - (1.0 - e)) / c ** 2


def _G(x, y, M: float):
    """``int_0^M exp(-x m) exp(-y (M - m)) dm`` for finite M (stable form)."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    lo = np.minimum(x, y)
    d = np.abs(x - y)
    with np.errstate(divide="ignore", invalid="ignore"):
        core = np.where(d * M > 1e-12, -np.expm1(-d * M) / np.where(d > 0, d, 1.0), M)
    return np.exp(-lo * M) * core


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
    layer: int
    symbol: str
    name: str
    energy_kev: float
    mass_fraction: float
    cs: np.ndarray          # Q_j(E) on the SF grid
    mu: np.ndarray          # mu_c(E_j) for every layer c


class MatrixModel:
    """Primary and secondary fluorescence sensitivities for a layer stack.

    Parameters
    ----------
    sample : Layer or Stack
        A single :class:`Layer` is treated as a one-layer stack.
    geometry : Geometry
    energy_kev, flux_01bw : (n,) arrays
        Incident spectrum, **per 0.1 % bandwidth** as returned by
        ``xrf_beamline.Beamline.spectrum``; converted to per keV internally.
        A uniform grid with ~10 eV steps is appropriate.
    secondary : bool
        Include secondary fluorescence.
    target : int, optional
        Index of the layer being quantified (required for a multi-layer
        stack).  Per-layer quantities default to it, and elements that are
        not part of the stack (contaminants) are treated as if in it.
    sf_step_kev : float
        The secondary term is smooth in E and is evaluated on a grid decimated
        to about this step (default 40 eV).
    n_u : int
        Points of the logarithmic direction-cosine grid for the SF kernel.
    exciter_min_rel : float
        Exciting lines weaker than this fraction of their element's strongest
        line (primary production in its layer) are ignored.
    """

    def __init__(self, sample: Union[Layer, Stack], geometry: Geometry,
                 energy_kev: np.ndarray, flux_01bw: np.ndarray,
                 secondary: bool = True, target: Optional[int] = None,
                 sf_step_kev: float = 0.04, n_u: int = 160,
                 exciter_min_rel: float = 1e-3):
        self.stack = sample if isinstance(sample, Stack) else Stack([sample])
        if target is None:
            if len(self.stack) > 1:
                raise ValueError("A multi-layer stack needs the index of the target layer.")
            target = 0
        if not 0 <= target < len(self.stack):
            raise ValueError(f"target {target} outside the stack of {len(self.stack)} layers.")
        self.target = int(target)
        self.geometry = geometry
        self.energy = np.asarray(energy_kev, dtype=float)
        self.flux_01bw = np.asarray(flux_01bw, dtype=float)
        self.density = per_unit_energy(self.energy, self.flux_01bw)
        self.secondary_enabled = bool(secondary)
        self.sf_step_kev = sf_step_kev
        self.n_u = int(n_u)
        self.exciter_min_rel = exciter_min_rel

        g = geometry
        self._M = np.array([lay.mass_thickness for lay in self.stack])
        self._mu_E = [lay.mu(self.energy) for lay in self.stack]
        step = float(np.median(np.diff(self.energy)))
        self._dec = max(1, int(round(sf_step_kev / step)))
        self._E_sf = self.energy[::self._dec]
        self._phi_sf = self.density[::self._dec]
        self._mu_sf = [m[::self._dec] for m in self._mu_E]
        # incident-beam transmission through the layers above each layer
        self._tin = [np.exp(-sum(self._mu_E[c] * self._M[c] for c in range(k)) / g.sin_in)
                     if k else np.ones_like(self.energy) for k in range(len(self.stack))]
        self._tin_sf = [t[::self._dec] for t in self._tin]
        s = np.linspace(np.log(1e-7), 0.0, self.n_u)
        self._s, self._u = s, np.exp(s)
        self._exciters: Optional[List[_Exciter]] = None

    # ------------------------------------------------------------------ setup

    @property
    def layer(self) -> Layer:
        """The target layer."""
        return self.stack[self.target]

    def with_layer(self, layer: Layer, index: Optional[int] = None) -> "MatrixModel":
        """Same beam, geometry, numerics and target, one layer replaced
        (the target layer by default)."""
        idx = self.target if index is None else index
        return MatrixModel(self.stack.replace(idx, layer), self.geometry, self.energy,
                           self.flux_01bw, secondary=self.secondary_enabled, target=self.target,
                           sf_step_kev=self.sf_step_kev, n_u=self.n_u,
                           exciter_min_rel=self.exciter_min_rel)

    def with_stack(self, stack: Union[Layer, Stack], target: Optional[int] = None) -> "MatrixModel":
        return MatrixModel(stack, self.geometry, self.energy, self.flux_01bw,
                           secondary=self.secondary_enabled,
                           target=self.target if target is None else target,
                           sf_step_kev=self.sf_step_kev, n_u=self.n_u,
                           exciter_min_rel=self.exciter_min_rel)

    @property
    def exciters(self) -> List[_Exciter]:
        """Lines of the stack's own elements that can drive secondary fluorescence."""
        if self._exciters is None:
            ex: List[_Exciter] = []
            sin_in = self.geometry.sin_in
            for a, lay in enumerate(self.stack):
                for z, w in lay.mass_fractions.items():
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
                        strength = trapezoid(self._phi_sf / sin_in * cs * self._tin_sf[a], self._E_sf)
                        cands.append((name[:-5], e_j, cs, strength))
                    if not cands:
                        continue
                    smax = max(c[3] for c in cands)
                    for name, e_j, cs, st in cands:
                        if st >= self.exciter_min_rel * smax:
                            mu = np.array([l.mu(e_j) for l in self.stack])
                            ex.append(_Exciter(a, sym, name, e_j, w, cs, mu))
            self._exciters = ex
            logger.info("%d exciting lines in %s", len(ex), self.stack)
        return self._exciters

    def _k(self, layer: Optional[int]) -> int:
        return self.target if layer is None else int(layer)

    def _b(self, line_energy_kev: float) -> np.ndarray:
        return np.array([lay.mu(line_energy_kev) for lay in self.stack]) / self.geometry.sin_out

    def _tout(self, b: np.ndarray, k: int) -> float:
        return float(np.exp(-np.sum(b[:k] * self._M[:k]))) if k else 1.0

    # ------------------------------------------------------------ physics

    def primary(self, z: int, macro: int, line_energy_kev: float,
                layer: Optional[int] = None) -> float:
        """``P_il^k`` — primary sensitivity per unit mass fraction in layer ``k``
        (the target layer by default), overlayer attenuation included."""
        k = self._k(layer)
        g = self.geometry
        q = _cs_fluor(z, macro, self.energy)
        if not q.any():
            return 0.0
        b = self._b(line_energy_kev)
        a_k = self._mu_E[k] / g.sin_in
        val = trapezoid(self.density / g.sin_in * q * self._tin[k] * _F(a_k + b[k], self._M[k]),
                        self.energy)
        return float(val) * self._tout(b, k)

    def secondary(self, z: int, macro: int, line_energy_kev: float,
                  layer: Optional[int] = None, by_exciter: bool = False):
        """``S_il^k`` — secondary sensitivity per unit mass fraction in layer
        ``k`` (target by default), summed over exciting lines in every layer.

        With ``by_exciter=True`` also returns ``{"<layer>:<element> <line>": value}``.
        """
        if not self.secondary_enabled:
            return (0.0, {}) if by_exciter else 0.0
        k = self._k(layer)
        g = self.geometry
        M = self._M
        b = self._b(line_energy_kev)
        u, s = self._u, self._s
        total, parts = 0.0, {}
        for ex in self.exciters:
            q_ij = _cs_fluor_scalar(z, macro, ex.energy_kev)
            if q_ij <= 0.0:
                continue
            a = ex.layer
            kc = ex.mu[:, None] / u[None, :]                    # (n_layers, n_u)
            A = (self._mu_sf[a] / g.sin_in)[:, None]            # (n_E, 1)
            if a == k:
                kern = trapezoid(_D(A, b[k], kc[k][None, :], M[k]), s, axis=1)
            elif a < k:
                between = np.exp(-sum(kc[c] * M[c] for c in range(a + 1, k))) if k - a > 1 else 1.0
                tail = between * _F(b[k] + kc[k], M[k])          # (n_u,)
                kern = trapezoid(_G(A, kc[a][None, :], M[a]) * tail[None, :], s, axis=1)
            else:
                between = np.exp(-sum(kc[c] * M[c] for c in range(k + 1, a))) if a - k > 1 else 1.0
                tail = between * _G(b[k], kc[k], M[k])           # (n_u,)
                kern = trapezoid(_F(A + kc[a][None, :], M[a]) * tail[None, :], s, axis=1)
            val = 0.5 * q_ij * ex.mass_fraction * float(trapezoid(
                self._phi_sf / g.sin_in * ex.cs * self._tin_sf[a] * kern, self._E_sf))
            val *= self._tout(b, k)
            total += val
            key = f"{a}:{ex.symbol} {ex.name}"
            parts[key] = parts.get(key, 0.0) + val
        return (total, parts) if by_exciter else total

    def sensitivity(self, z: int, macro: int, line_energy_kev: float,
                    layer: Optional[int] = None) -> float:
        """``P_il^k + S_il^k`` (per unit mass fraction in layer ``k``)."""
        p = self.primary(z, macro, line_energy_kev, layer)
        if p <= 0.0:
            return 0.0
        return p + self.secondary(z, macro, line_energy_kev, layer)

    def emission(self, z: int, macro: int, line_energy_kev: float,
                 secondary: bool = True) -> float:
        """Line intensity from the whole stack, ``sum_k w_i^k (P^k [+ S^k])``.

        Elements absent from the stack (contaminants) are evaluated per unit
        mass fraction in the target layer.  This is the weighting that sets
        relative intensities *within* a line group.
        """
        layers = [(k, lay.mass_fractions[z]) for k, lay in enumerate(self.stack)
                  if z in lay.mass_fractions]
        if not layers:
            layers = [(self.target, 1.0)]
        tot = 0.0
        for k, w in layers:
            p = self.primary(z, macro, line_energy_kev, k)
            if p > 0 and secondary:
                p += self.secondary(z, macro, line_energy_kev, k)
            tot += w * p
        return tot

    def enhancement(self, z: int, macro: int, line_energy_kev: float,
                    layer: Optional[int] = None) -> float:
        """Secondary / primary, for layer ``k`` (target by default)."""
        p = self.primary(z, macro, line_energy_kev, layer)
        return self.secondary(z, macro, line_energy_kev, layer) / p if p > 0 else 0.0

    def absorption_factor(self, z: int, macro: int, line_energy_kev: float,
                          layer: Optional[int] = None) -> float:
        """Primary intensity of layer ``k`` relative to the same layer free of
        absorption (thin-film limit, no overlayers): 1 means no absorption.
        Includes the overlayers.  NaN for a bulk layer."""
        k = self._k(layer)
        if not np.isfinite(self._M[k]):
            return float("nan")
        g = self.geometry
        q = _cs_fluor(z, macro, self.energy)
        thin = trapezoid(self.density / g.sin_in * q * self._M[k], self.energy)
        return self.primary(z, macro, line_energy_kev, k) / thin if thin > 0 else float("nan")


# =============================================================================
# Quantification by iteration on the composition of the target layer
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
    """Self-consistent composition of the target layer from fitted reference-line areas.

    For every group ``g`` the measured reference-line area is modelled as

        A_g  ∝  w_g^t s_g^t + C_g,     s_g^k = (P^k + S^k) eff,
        C_g  = sum_{k != t} w_g^k s_g^k     (known layers: caps, substrate)

    so, relative to the reference group ``r``,

        w_i^t = [ (A_i / A_r) (w_r^t s_r^t + C_r) - C_i ] / s_i^t

    The target composition (atoms per reference atom, plus ``fixed_atoms``
    for unmeasured elements) is rebuilt from these mass fractions and the
    sensitivities recomputed until the ratios change by less than ``tol``.
    Density and thicknesses are held fixed.  Elements of the target layer
    that are neither analytes, reference nor fixed are dropped, because the
    absolute mass fraction of the reference is needed once other layers
    contribute.  For a one-layer sample ``C = 0`` and this reduces to
    ``n_i/n_r = (A_i/A_r) (s_r W_r)/(s_i W_i)``.

    Parameters
    ----------
    areas : {label: reference-line area}
    groups : {label: LineGroup}
    matrix : MatrixModel
        Its target layer supplies the starting composition.
    detector : xrf_fit.Detector
    reference : str
        Label of the reference group, e.g. ``"As K"``.
    analytes : sequence of str
    fixed_atoms : {symbol: atoms per reference atom}, optional

    Returns
    -------
    dict with keys ``ratios`` ({symbol: n/n_ref} in the target layer),
    ``layer`` and ``matrix`` (converged), ``k`` ({label: (n_i/n_r)/(A_i/A_r)}),
    ``enhancement`` and ``absorption`` (target layer, reference lines),
    ``other_layers`` ({label: fraction of the modelled intensity coming from
    the other layers}), ``terms`` ({label: (s_t, C)}), ``iterations``,
    ``history``.
    """
    fixed_atoms = dict(fixed_atoms or {})
    t = matrix.target
    g_ref = groups[reference]
    ref_el = g_ref.element
    z_ref = xraylib.SymbolToAtomicNumber(ref_el)
    w_ref_at = xraylib.AtomicWeight(z_ref)

    def terms(mm, g):
        z, macro, e = _line(g)
        eff = detector.efficiency(e * 1000.0)
        s_t = mm.sensitivity(z, macro, e, layer=t) * eff
        c = sum(lay.mass_fractions.get(z, 0.0) * mm.sensitivity(z, macro, e, layer=k)
                for k, lay in enumerate(mm.stack) if k != t and z in lay.mass_fractions) * eff
        return s_t, c

    atoms0 = matrix.layer.atoms
    n0 = atoms0.get(ref_el, 1.0)
    ratios = {el: n / n0 for el, n in atoms0.items()}
    history = [dict(ratios)]
    mm = matrix
    it = 0
    for it in range(1, max_iter + 1):
        w = mm.layer.mass_fractions
        w_r = w.get(z_ref, 0.0)
        if w_r <= 0:
            raise ValueError(f"reference element {ref_el} is not in the target layer.")
        s_r, c_r = terms(mm, g_ref)
        i_ref = w_r * s_r + c_r
        new = {ref_el: 1.0}
        new.update(fixed_atoms)
        for lab in analytes:
            g = groups[lab]
            s_i, c_i = terms(mm, g)
            w_i = (areas[lab] / areas[reference] * i_ref - c_i) / s_i
            if w_i < 0:
                logger.warning("%s: the other layers account for more than the measured "
                               "intensity; clipped to 0.", lab)
                w_i = 0.0
            w_at = xraylib.AtomicWeight(xraylib.SymbolToAtomicNumber(g.element))
            new[g.element] = float((w_i / w_at) / (w_r / w_ref_at))
        change = max(abs(new.get(el, 0.0) - ratios.get(el, 0.0)) for el in set(new) | set(ratios))
        ratios = new
        history.append(dict(ratios))
        mm = mm.with_layer(mm.layer.with_composition(ratios))
        if change < tol:
            break
    else:
        logger.warning("quantify_layer did not converge in %d iterations", max_iter)

    enh, absf, other, k_fac, all_terms = {}, {}, {}, {}, {}
    w = mm.layer.mass_fractions
    for lab in [reference, *analytes]:
        g = groups[lab]
        z, macro, e = _line(g)
        enh[lab] = mm.enhancement(z, macro, e)
        absf[lab] = mm.absorption_factor(z, macro, e)
        s_t, c = terms(mm, g)
        all_terms[lab] = (s_t, c)
        tot = w.get(z, 0.0) * s_t + c
        other[lab] = c / tot if tot > 0 else 0.0
        if lab != reference:
            k_fac[lab] = ratios[g.element] / (areas[lab] / areas[reference])
    return dict(ratios=ratios, layer=mm.layer, matrix=mm, k=k_fac, enhancement=enh,
                absorption=absf, other_layers=other, terms=all_terms,
                reference=reference, iterations=it, history=history)


def atom_ratio(quant: dict, label: str, area, area_ref, err=None, err_ref=None):
    """Per-pixel atom ratio ``n_i / n_ref`` in the target layer, using the
    converged sensitivities of :func:`quantify_layer` (matrix held fixed).

    ``n_i/n_r = [(R I_r - C_i)/s_i / W_i] / (w_r / W_r)`` with ``R = A_i/A_r``,
    ``I_r = w_r s_r + C_r``.  Reduces to :func:`xrf_fit.number_ratio` for a
    one-layer sample.  Errors are propagated from the two areas, uncorrelated.

    Returns
    -------
    ratio, ratio_err (None without errors)
    """
    ref = quant["reference"]
    lay = quant["layer"]
    z_ref = xraylib.SymbolToAtomicNumber(_ref_symbol(quant))
    w_r = lay.mass_fractions[z_ref]
    s_r, c_r = quant["terms"][ref]
    s_i, c_i = quant["terms"][label]
    el = _label_symbol(label)
    w_at_i = xraylib.AtomicWeight(xraylib.SymbolToAtomicNumber(el))
    w_at_r = xraylib.AtomicWeight(z_ref)
    a = np.asarray(area, dtype=float)
    b = np.asarray(area_ref, dtype=float)
    i_r = w_r * s_r + c_r
    scale = w_at_r / (w_at_i * w_r * s_i)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = a / b
        ratio = (R * i_r - c_i) * scale
        err_out = None
        if err is not None and err_ref is not None:
            dR = np.abs(R) * np.sqrt((np.asarray(err) / a) ** 2 + (np.asarray(err_ref) / b) ** 2)
            err_out = dR * i_r * scale
    return ratio, err_out


def _label_symbol(label: str) -> str:
    return label.split()[0]


def _ref_symbol(quant: dict) -> str:
    return _label_symbol(quant["reference"])
