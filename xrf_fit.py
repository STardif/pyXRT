"""
Module: xrf_fit.py

Constrained fitting of synchrotron X-ray fluorescence spectra (BM32, ESRF).

The model
---------
A spectrum is described as

    y(c) = B(c) + sum_g  A_g * sum_{l in g} r_l * V(E(c) - E_l ; sigma(E_l), gamma)

where

* ``c``        is the MCA channel and ``E(c)`` the energy calibration,
* ``B(c)``     is a polynomial background,
* ``g``        runs over *line groups* — one group per (element, shell family),
                 e.g. ("Nd", "L"), ("Cu", "K"),
* ``A_g``      is the single free amplitude of group ``g``: the integrated area of
                 the group's **most intense** line,
* ``r_l``      is the tabulated intensity of line ``l`` **relative to that most
                 intense line** (so ``r = 1`` for the reference line),
* ``V``        is a unit-area Voigt profile,
* ``sigma(E)`` follows the detector resolution law (two global parameters),
* ``gamma``    is one global Lorentzian width.

So an N-element fit has ``N + n_bkg + 3`` free parameters instead of four per
line.  Peak positions are never free: they come from the tabulated line energies
and the calibration.

Relative intensities: why not RadRate
-------------------------------------
``xraylib.RadRate`` is normalised **within a subshell** — the L1, L2 and L3 rates
each sum to 1 independently — so it carries no information about the relative
ionisation of those subshells.  Using it across an L series is wrong: for Ir it
predicts Lb1/La1 = 1.13, where the Kissel partial fluorescence cross-sections
give 0.65 and the measured ratio on the NIO295 map is 0.77.

This module therefore takes relative intensities from
``xraylib.CS_FluorLine_Kissel(Z, line, E0)``, which includes subshell ionisation
cross-sections, fluorescence yields and Coster-Kronig transfer.  For a K series
the two approaches agree, since all K lines share one initial vacancy.

``E0`` is a single *effective* excitation energy.  Under white-beam illumination
the physically correct weighting is the flux-weighted integral
``int sigma(E) Phi(E) dE``; a scalar ``E0`` is a deliberate first-order stand-in
until the beamline spectrum model is wired in.  ``E0`` must exceed the highest
relevant absorption edge (e.g. the Ir L1 edge at 13.42 keV) or the high-subshell
lines are silently given zero intensity.

What this module does *not* do yet
----------------------------------
* No detector efficiency: a line's modelled intensity is its emitted intensity.
  Comparing groups at different energies (K vs L, or two elements far apart) is
  therefore not quantitative.
* No self-absorption / matrix correction, no secondary fluorescence.
* No escape or pile-up peaks.  On the NIO295 map the dead-time fraction reaches
  0.79 and a 2x(Ir La) sum peak is visible near channel 1830, so pile-up is
  present and will appear in the residual.
* The energy calibration is taken as given, not fitted.

Authors: Samuel Tardif, with Claude (Anthropic).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import xraylib
from scipy.integrate import simpson
from scipy.optimize import curve_fit
from scipy.special import voigt_profile

logger = logging.getLogger(__name__)

# xraylib raises ValueError for lines/edges that do not exist for a given Z;
# silence its stderr chatter and handle the exceptions instead.
xraylib.SetErrorMessages(0)

__all__ = [
    "Calibration",
    "Detector",
    "Line",
    "LineGroup",
    "XRFModel",
    "FitResult",
    "element_line_groups",
    "normalise_spectra",
    "average_spectrum",
    "counting_error",
    "snip_background",
    "fit_average_spectrum",
    "plot_fit",
    "identify_peaks",
    "MapResult",
    "design_matrix",
    "fit_map",
    "pixel_parameters",
    "select_pixels",
    "plot_pixel_fits",
]


# =============================================================================
# Energy calibration
# =============================================================================

@dataclass(frozen=True)
class Calibration:
    """Linear MCA channel <-> energy calibration, ``E[eV] = gain * channel + offset``.

    The BM32 XIA default below was determined from Ga/As K lines; it is used here
    as a fixed input.  Fitting it from identified lines is a separate concern.

    Parameters
    ----------
    gain : float
        eV per channel.
    offset : float
        Energy of channel 0, in eV.
    """

    gain: float = 10.032
    offset: float = -31.263

    def energy(self, channel: np.ndarray | float) -> np.ndarray | float:
        """Channel -> energy (eV)."""
        return self.gain * np.asarray(channel, dtype=float) + self.offset

    def channel(self, energy_ev: np.ndarray | float) -> np.ndarray | float:
        """Energy (eV) -> channel (not rounded)."""
        return (np.asarray(energy_ev, dtype=float) - self.offset) / self.gain

    def channel_range(self, emin_ev: float, emax_ev: float, n_channels: int) -> Tuple[int, int]:
        """Half-open channel range ``[lo, hi)`` covering an energy window."""
        lo = max(0, int(np.floor(self.channel(emin_ev))))
        hi = min(n_channels, int(np.ceil(self.channel(emax_ev))) + 1)
        return lo, hi


# =============================================================================
# Detector response
# =============================================================================

#: Mean energy to create one electron-hole pair, in eV.
PAIR_ENERGY_EV = {"Si": 3.85, "Ge": 2.96}


@dataclass
class Detector:
    """Energy-resolution model of a solid-state detector.

    The Gaussian width combines energy-independent electronic noise with the
    Fano-limited statistical term::

        sigma(E)^2 = noise_ev^2 + fano * pair_energy * E
        FWHM(E)    = 2.3548 * sigma(E)

    Defaults correspond to ~135 eV FWHM at 5.9 keV on a Si detector, i.e. the
    middle of the 125-150 eV range quoted for the BM32 XIA.

    Parameters
    ----------
    material : {"Si", "Ge"}
        Sets the electron-hole pair creation energy.
    noise_ev : float
        Electronic-noise contribution to sigma, in eV.
    fano : float
        Fano factor (~0.12 for Si, ~0.06-0.13 for Ge).
    gamma_ev : float
        Lorentzian HWHM of the Voigt profile, in eV.  Physically this is the
        natural level width (a few eV for an L line); in practice it also
        absorbs part of the low-energy tail, so treat a fitted value above
        ~30 eV as a sign that a proper tail term is needed.
    """

    material: str = "Si"
    noise_ev: float = 30.0
    fano: float = 0.12
    gamma_ev: float = 5.0

    @property
    def pair_energy_ev(self) -> float:
        try:
            return PAIR_ENERGY_EV[self.material]
        except KeyError:
            raise ValueError(f"material must be one of {list(PAIR_ENERGY_EV)}, got {self.material!r}")

    def sigma_ev(self, energy_ev: np.ndarray | float,
                 noise_ev: Optional[float] = None,
                 fano: Optional[float] = None) -> np.ndarray | float:
        """Gaussian sigma (eV) at one or more energies."""
        n = self.noise_ev if noise_ev is None else noise_ev
        f = self.fano if fano is None else fano
        return np.sqrt(n ** 2 + f * self.pair_energy_ev * np.asarray(energy_ev, dtype=float))

    def fwhm_ev(self, energy_ev: np.ndarray | float, **kw) -> np.ndarray | float:
        """Gaussian FWHM (eV) at one or more energies."""
        return 2.3548 * self.sigma_ev(energy_ev, **kw)

    @classmethod
    def from_fwhm(cls, fwhm_ev: float, at_energy_ev: float, material: str = "Si",
                  fano: float = 0.12, gamma_ev: float = 5.0) -> "Detector":
        """Build a detector whose FWHM equals ``fwhm_ev`` at ``at_energy_ev``.

        Solves for the electronic-noise term given a Fano factor, which is the
        usual way a quoted resolution figure ("135 eV at Mn Ka") is turned into
        a resolution law.
        """
        sigma = fwhm_ev / 2.3548
        var_fano = fano * PAIR_ENERGY_EV[material] * at_energy_ev
        var_noise = sigma ** 2 - var_fano
        if var_noise <= 0:
            raise ValueError(
                f"FWHM {fwhm_ev} eV at {at_energy_ev} eV is below the Fano limit "
                f"({2.3548 * np.sqrt(var_fano):.1f} eV) for {material} with F={fano}."
            )
        return cls(material=material, noise_ev=float(np.sqrt(var_noise)), fano=fano, gamma_ev=gamma_ev)


# =============================================================================
# Fluorescence lines
# =============================================================================

@dataclass(frozen=True)
class Line:
    """One fluorescence line.

    Attributes
    ----------
    name : str
        xraylib macro name without the ``_LINE`` suffix, e.g. ``"LA1"``.
    energy_ev : float
        Tabulated line energy in eV.
    rel : float
        Intensity relative to the most intense line of its group (reference
        line has ``rel == 1``).
    """

    name: str
    energy_ev: float
    rel: float


@dataclass
class LineGroup:
    """A set of lines sharing one free amplitude.

    One group per (element, shell family).  ``lines[0]`` is the reference line,
    whose integrated area is the fitted parameter.
    """

    element: str
    family: str          # "K", "L" or "M"
    lines: List[Line]

    @property
    def label(self) -> str:
        return f"{self.element} {self.family}"

    @property
    def reference(self) -> Line:
        return self.lines[0]

    @property
    def total_rel(self) -> float:
        """Sum of relative intensities — converts the fitted reference-line area
        into the total area of the modelled group."""
        return float(sum(l.rel for l in self.lines))


#: Shell family of a line, from the initial-vacancy shell in its xraylib name.
def _family_of(macro_name: str) -> Optional[str]:
    base = macro_name[:-5] if macro_name.endswith("_LINE") else macro_name
    if base.startswith("K"):
        return "K"
    if base.startswith("L"):
        return "L"
    if base.startswith("M"):
        return "M"
    return None


def _all_line_macros() -> Dict[str, int]:
    """All ``*_LINE`` macros exposed by xraylib, keyed by name."""
    return {n: getattr(xraylib, n) for n in dir(xraylib) if n.endswith("_LINE")}


_LINE_MACROS = _all_line_macros()

#: xraylib exposes every transition twice: once under its Siegbahn name
#: (``KA1``, ``LB2``, ``LG1``, ``LL`` …) and once under its IUPAC name
#: (``KL3``, ``L3N5``, ``L2N4``, ``L3M1`` …), plus composite macros (``KA`` =
#: KL3+KL2, ``LA`` = LA1+LA2).  Enumerating all of them would double- or
#: triple-count intensity, so we keep only the IUPAC set: it is complete,
#: non-redundant, and one name per physical transition.
#:
#: The pattern demands a full subshell designator on both sides, which is what
#: separates IUPAC from Siegbahn: ``L3M5`` matches, ``LL`` (= L3M1) does not,
#: because its initial shell lacks a subshell index.
_IUPAC_LINE_RE = re.compile(r"^(K|L[1-3]|M[1-5])(K|L[1-3]|M[1-5]|N[1-7]|O[1-7]|P[1-5])$")


def _is_iupac_line(macro_name: str) -> bool:
    base = macro_name[:-5] if macro_name.endswith("_LINE") else macro_name
    return bool(_IUPAC_LINE_RE.match(base))


def _line_energy_ev(z: int, macro: int) -> Optional[float]:
    try:
        e = xraylib.LineEnergy(z, macro)
    except ValueError:
        return None
    return e * 1000.0 if e > 0 else None


def _line_strength_spectral(z: int, macro: int,
                            energy_kev: np.ndarray, flux: np.ndarray) -> float:
    """Flux-weighted line strength, ``integral sigma_l(E) * Phi(E) dE``.

    The correct relative-intensity measure under polychromatic illumination:
    each subshell is excited only by the part of the spectrum above its own
    edge, so subshells whose edges sit inside the band are down-weighted
    relative to what a single effective energy would give.

    Integrated with Simpson's rule on the supplied grid.  Adaptive quadrature is
    the wrong tool here: the integrand is identically zero below the edge and
    jumps discontinuously above it.

    Parameters
    ----------
    energy_kev, flux : (n,) arrays
        Incident spectrum.  Only the shape matters; any constant scale cancels.
    """
    cs = np.zeros_like(energy_kev)
    for i, ek in enumerate(energy_kev):
        try:
            v = xraylib.CS_FluorLine_Kissel(z, macro, float(ek))
        except ValueError:
            continue                       # below the edge
        if v > 0:
            cs[i] = v
    if not np.any(cs):
        return 0.0
    return float(max(0.0, simpson(cs * flux, x=energy_kev)))


def _line_strength(z: int, macro: int, e0_kev: float, weighting: str) -> float:
    """Unnormalised line intensity.

    ``"kissel"`` uses the partial fluorescence cross-section at ``e0_kev``
    (cm^2/g), which is comparable across subshells.  ``"radrate"`` uses the
    radiative rate, which is only comparable *within* a subshell.
    """
    try:
        if weighting == "kissel":
            return max(0.0, xraylib.CS_FluorLine_Kissel(z, macro, e0_kev))
        if weighting == "radrate":
            return max(0.0, xraylib.RadRate(z, macro))
    except ValueError:
        return 0.0
    raise ValueError(f"weighting must be 'kissel' or 'radrate', got {weighting!r}")


def element_line_groups(
    elements: Sequence[str | int],
    emin_ev: float,
    emax_ev: float,
    e0_eff_kev: float = 20.0,
    families: Sequence[str] = ("K", "L", "M"),
    min_rel: float = 0.01,
    weighting: str = "kissel",
    merge_within_ev: float = 0.0,
    incident_spectrum: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> List[LineGroup]:
    """Build the line groups to fit.

    Parameters
    ----------
    elements
        Element symbols (``"Nd"``) or atomic numbers (``60``).
    emin_ev, emax_ev
        Only lines inside this energy window are kept — normally the fit ROI.
    e0_eff_kev
        Effective excitation energy for the Kissel cross-sections.  Must exceed
        the highest edge you care about; see the module docstring.
    families
        Shell families to consider.
    min_rel
        Discard lines weaker than this fraction of their group's reference line.
    weighting
        ``"kissel"`` (default, correct across subshells) or ``"radrate"``.
        Ignored when ``incident_spectrum`` is given.
    incident_spectrum : (energy_kev, flux), optional
        Incident white-beam spectrum.  When supplied, line strengths become the
        flux-weighted integrals ``int sigma_l(E) Phi(E) dE`` instead of
        cross-sections at the single energy ``e0_eff_kev``.  This is the correct
        treatment for polychromatic illumination and is what changes the L
        subshell weights of heavy elements, whose edges fall inside the band.
        Only the shape of ``flux`` matters.  Build one with
        ``xrf_beamline.BM32_WHITE_BEAM().spectrum(energy_grid)``.
    merge_within_ev
        If > 0, lines closer than this are merged into one effective line at
        their intensity-weighted mean energy.  Useful to collapse unresolvable
        multiplets (the detector FWHM is ~130 eV) and shrink the model.

    Returns
    -------
    list of LineGroup
        One per (element, family) that has at least one surviving line, ordered
        as given in ``elements``.  Each group's ``lines`` are sorted by
        descending relative intensity, so ``lines[0]`` is the reference.
    """
    groups: List[LineGroup] = []

    for item in elements:
        if isinstance(item, str):
            try:
                z = xraylib.SymbolToAtomicNumber(item.capitalize())
            except ValueError:
                logger.warning("Unknown element symbol %r — skipped.", item)
                continue
        else:
            z = int(item)
        symbol = xraylib.AtomicNumberToSymbol(z)

        for fam in families:
            found: List[Tuple[str, float, float]] = []   # (name, energy, strength)

            for name, macro in _LINE_MACROS.items():
                if not _is_iupac_line(name) or _family_of(name) != fam:
                    continue
                e_ev = _line_energy_ev(z, macro)
                if e_ev is None or not (emin_ev <= e_ev <= emax_ev):
                    continue
                if incident_spectrum is None:
                    s = _line_strength(z, macro, e0_eff_kev, weighting)
                else:
                    s = _line_strength_spectral(z, macro, *incident_spectrum)
                if s <= 0:
                    continue
                found.append((name[:-5], e_ev, s))

            if not found:
                continue

            if merge_within_ev > 0:
                found = _merge_close_lines(found, merge_within_ev)

            s_max = max(s for _, _, s in found)
            lines = [Line(name=n, energy_ev=e, rel=s / s_max)
                     for n, e, s in found if s / s_max >= min_rel]
            lines.sort(key=lambda l: -l.rel)

            if lines:
                groups.append(LineGroup(element=symbol, family=fam, lines=lines))
                logger.info("%s %s: %d lines, reference %s at %.0f eV",
                            symbol, fam, len(lines), lines[0].name, lines[0].energy_ev)

    return groups


def _merge_close_lines(found: List[Tuple[str, float, float]],
                       tol_ev: float) -> List[Tuple[str, float, float]]:
    """Collapse lines within ``tol_ev`` into intensity-weighted effective lines."""
    found = sorted(found, key=lambda t: t[1])
    merged: List[Tuple[str, float, float]] = []
    bucket = [found[0]]
    for entry in found[1:]:
        if entry[1] - bucket[-1][1] <= tol_ev:
            bucket.append(entry)
        else:
            merged.append(_collapse(bucket))
            bucket = [entry]
    merged.append(_collapse(bucket))
    return merged


def _collapse(bucket: List[Tuple[str, float, float]]) -> Tuple[str, float, float]:
    s_tot = sum(s for _, _, s in bucket)
    if s_tot <= 0:
        return bucket[0]
    e_eff = sum(e * s for _, e, s in bucket) / s_tot
    name = bucket[int(np.argmax([s for _, _, s in bucket]))][0]
    if len(bucket) > 1:
        name = f"{name}+{len(bucket) - 1}"
    return (name, e_eff, s_tot)


# =============================================================================
# Normalisation and averaging
# =============================================================================

def normalise_spectra(
    spectra: np.ndarray,
    monitor: Optional[np.ndarray] = None,
    live_time: Optional[np.ndarray] = None,
    normalise_monitor: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Correct raw MCA spectra for incident flux and detector dead time.

    Both corrections are multiplicative per pixel::

        y_corrected = y_raw / (monitor_rel * live_time)

    ``live_time`` is the *active* counting time reported by the XIA
    (``xia_det0_live_time``), so dividing by it converts recorded counts into a
    true incoming rate.  This is the correction that matters most on BM32 white-beam
    maps: the dead-time fraction is strongly position dependent (0.33-0.79 on the
    NIO295 map, i.e. a factor 1.5-4.8 in the correction) because it tracks total
    scattering power, so it does **not** cancel in an elemental ratio.

    ``monitor`` is divided by its own mean when ``normalise_monitor`` is True, so
    the output keeps the magnitude of counts-per-live-second rather than becoming
    counts per monitor count.

    Parameters
    ----------
    spectra : (n_pixels, n_channels) array
    monitor : (n_pixels,) array, optional
    live_time : (n_pixels,) array, optional
        In seconds.
    normalise_monitor : bool
        Divide the monitor by its mean before use.

    Returns
    -------
    corrected : (n_pixels, n_channels) float array
        Pixels with a non-positive or non-finite scale are set to NaN.
    scale : (n_pixels,) float array
        The divisor actually applied.
    """
    spectra = np.asarray(spectra, dtype=float)
    n = spectra.shape[0]
    scale = np.ones(n, dtype=float)

    if monitor is not None:
        mon = np.asarray(monitor, dtype=float)
        if normalise_monitor:
            good = np.isfinite(mon) & (mon > 0)
            if not good.any():
                raise ValueError("monitor has no positive finite values.")
            mon = mon / mon[good].mean()
        scale *= mon

    if live_time is not None:
        scale *= np.asarray(live_time, dtype=float)

    bad = ~np.isfinite(scale) | (scale <= 0)
    if bad.any():
        logger.warning("%d/%d pixels have an invalid normalisation and are set to NaN.",
                       int(bad.sum()), n)
    safe = np.where(bad, np.nan, scale)
    return spectra / safe[:, None], scale


def average_spectrum(corrected: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Map-averaged spectrum and the standard error of that mean.

    The per-channel standard error across pixels is the right uncertainty for
    fitting an *average* spectrum: it propagates both counting statistics and
    genuine pixel-to-pixel variation, and needs no assumption about how the
    normalisation transformed the Poisson variance.

    Returns
    -------
    mean, sem : (n_channels,) arrays
    n_used : int
        Number of pixels contributing (NaN rows are dropped).
    """
    corrected = np.asarray(corrected, dtype=float)
    good = np.isfinite(corrected).all(axis=1)
    data = corrected[good]
    if data.shape[0] < 2:
        raise ValueError("Need at least 2 valid pixels to form a mean and its error.")
    mean = data.mean(axis=0)
    sem = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])
    return mean, sem, int(data.shape[0])


def counting_error(raw_spectra: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Poisson uncertainty of the normalised map-averaged spectrum.

    With ``y_avg = (1/N) * sum_i y_i / s_i`` and Poisson raw counts
    (``Var(y_i) = y_i``), the variance of the average is
    ``(1/N^2) * sum_i y_i / s_i^2``.

    This is the appropriate weight when fitting the *average* spectrum, and it is
    not the same as the standard error returned by :func:`average_spectrum`: the
    latter also contains genuine pixel-to-pixel variation, which is signal rather
    than noise as far as the average is concerned.  On a heterogeneous map the
    SEM can exceed this by an order of magnitude.

    Parameters
    ----------
    raw_spectra : (n_pixels, n_channels) array
        **Raw** counts, before normalisation.
    scale : (n_pixels,) array
        The divisor applied by :func:`normalise_spectra`.

    Returns
    -------
    (n_channels,) array
        1-sigma uncertainty, in the same units as the normalised average.
    """
    raw = np.asarray(raw_spectra, dtype=float)
    s = np.asarray(scale, dtype=float)
    good = np.isfinite(s) & (s > 0)
    if not good.any():
        raise ValueError("No pixel has a valid normalisation scale.")
    n = int(good.sum())
    return np.sqrt((raw[good] / s[good, None] ** 2).sum(axis=0)) / n


def snip_background(y: np.ndarray, width: int = 40, iterations: int = 24,
                    smooth: bool = True) -> np.ndarray:
    """SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping) background.

    The standard XRF continuum estimator (Ryan et al. 1988): iteratively clips
    each point to the mean of its neighbours at increasing then decreasing
    separation, which erodes peaks while following the slowly varying continuum.
    Operates on the log-log-square-root transformed spectrum so that the clipping
    is statistics-aware.

    Parameters
    ----------
    y : (n,) array
        Spectrum, assumed non-negative.
    width : int
        Maximum half-window in channels; should exceed the widest peak.
    iterations : int
        Number of decreasing-window passes after the increasing ones.
    smooth : bool
        Apply a 3-point moving average first, which stabilises the transform on
        low-count channels.

    Returns
    -------
    (n,) array
        Estimated background, same length as ``y``.
    """
    y = np.asarray(y, dtype=float).copy()
    if smooth:
        y = np.convolve(y, np.ones(3) / 3.0, mode="same")
    y = np.clip(y, 0.0, None)

    # LLS transform
    v = np.log(np.log(np.sqrt(y + 1.0) + 1.0) + 1.0)

    windows = list(range(1, width + 1)) + list(range(width, 0, -1))[:iterations]
    for w in windows:
        shifted_lo = np.roll(v, w)
        shifted_hi = np.roll(v, -w)
        shifted_lo[:w] = v[:w]
        shifted_hi[-w:] = v[-w:]
        v = np.minimum(v, 0.5 * (shifted_lo + shifted_hi))

    # inverse LLS
    return (np.exp(np.exp(v) - 1.0) - 1.0) ** 2 - 1.0


# =============================================================================
# The model
# =============================================================================

class XRFModel:
    """Constrained Voigt model of an XRF spectrum.

    Free parameters, in order:

    ====================  =========================================
    ``bkg_0 .. bkg_n``    polynomial background coefficients
    ``A_<group>``         reference-line area, one per line group
    ``noise_ev``          electronic-noise sigma term  (if fitted)
    ``fano``              Fano factor                  (if fitted)
    ``gamma_ev``          global Lorentzian HWHM        (if fitted)
    ====================  =========================================

    The background polynomial is evaluated in a centred, scaled channel
    coordinate so that its coefficients stay well conditioned.

    Parameters
    ----------
    groups : list of LineGroup
    calibration : Calibration
    detector : Detector
        Supplies the initial values (and the frozen values, if not fitted) of
        the resolution parameters.
    channels : (n,) array
        The channel grid of the fit ROI.
    background_order : int
        Polynomial order; 0 is a constant, 1 linear.  Use ``None`` for no
        background term (e.g. after SNIP stripping).
    fit_resolution : bool
        Free ``noise_ev`` and ``fano``.
    fit_gamma : bool
        Free ``gamma_ev``.
    """

    def __init__(self, groups: Sequence[LineGroup], calibration: Calibration,
                 detector: Detector, channels: np.ndarray,
                 background_order: Optional[int] = 1,
                 fit_resolution: bool = True, fit_gamma: bool = True):
        if not groups:
            raise ValueError("At least one line group is required.")
        self.groups = list(groups)
        self.calibration = calibration
        self.detector = detector
        self.channels = np.asarray(channels, dtype=float)
        self.background_order = background_order
        self.fit_resolution = bool(fit_resolution)
        self.fit_gamma = bool(fit_gamma)

        self.energies = calibration.energy(self.channels)

        # Centred/scaled abscissa for the background polynomial.
        c0, c1 = self.channels[0], self.channels[-1]
        self._u = 2.0 * (self.channels - c0) / max(c1 - c0, 1.0) - 1.0

        self.n_bkg = 0 if background_order is None else background_order + 1

        names = [f"bkg_{i}" for i in range(self.n_bkg)]
        names += [f"A_{g.label.replace(' ', '')}" for g in self.groups]
        if self.fit_resolution:
            names += ["noise_ev", "fano"]
        if self.fit_gamma:
            names += ["gamma_ev"]
        self.param_names: List[str] = names

    # -- parameter unpacking ------------------------------------------------

    def _unpack(self, params: Sequence[float]):
        i = 0
        bkg = np.asarray(params[i:i + self.n_bkg], dtype=float); i += self.n_bkg
        amps = np.asarray(params[i:i + len(self.groups)], dtype=float); i += len(self.groups)
        if self.fit_resolution:
            noise, fano = float(params[i]), float(params[i + 1]); i += 2
        else:
            noise, fano = self.detector.noise_ev, self.detector.fano
        gamma = float(params[i]) if self.fit_gamma else self.detector.gamma_ev
        return bkg, amps, noise, fano, gamma

    # -- evaluation ---------------------------------------------------------

    def background(self, params: Sequence[float]) -> np.ndarray:
        bkg, *_ = self._unpack(params)
        if self.n_bkg == 0:
            return np.zeros_like(self.channels)
        return sum(c * self._u ** k for k, c in enumerate(bkg))

    def group_profile(self, index: int, params: Sequence[float]) -> np.ndarray:
        """Modelled counts of one line group (background excluded).

        Profiles are evaluated in *channel* space: ``voigt_profile`` integrates
        to 1 over its argument, so with widths expressed in channels the
        amplitude is directly the integrated number of counts summed over
        channels — the same quantity the data carries.
        """
        _, amps, noise, fano, gamma = self._unpack(params)
        g = self.groups[index]
        gain = self.calibration.gain
        gamma_ch = gamma / gain
        out = np.zeros_like(self.channels)
        for line in g.lines:
            sigma_ch = self.detector.sigma_ev(line.energy_ev, noise_ev=noise, fano=fano) / gain
            centre_ch = self.calibration.channel(line.energy_ev)
            out += (amps[index] * line.rel) * voigt_profile(
                self.channels - centre_ch, sigma_ch, gamma_ch)
        return out

    def __call__(self, channels: np.ndarray, *params: float) -> np.ndarray:
        """Total model, in the signature ``curve_fit`` expects."""
        total = self.background(params)
        for i in range(len(self.groups)):
            total = total + self.group_profile(i, params)
        return total

    # -- starting point -----------------------------------------------------

    def initial_guess(self, y: np.ndarray) -> Tuple[List[float], Tuple[List[float], List[float]]]:
        """Initial parameters and bounds, estimated from the data.

        Each group's amplitude is seeded from the observed height at its
        reference line, converted to an area with the modelled width; the
        background is seeded from the lower envelope of the ROI.
        """
        y = np.asarray(y, dtype=float)
        base = float(np.percentile(y, 5))

        p0: List[float] = []
        lo: List[float] = []
        hi: List[float] = []

        if self.n_bkg:
            p0 += [base] + [0.0] * (self.n_bkg - 1)
            lo += [-np.inf] * self.n_bkg
            hi += [np.inf] * self.n_bkg

        span = max(float(y.max()) - base, 1.0)
        for g in self.groups:
            ref = g.reference
            ch = self.calibration.channel(ref.energy_ev)
            j = int(np.clip(np.searchsorted(self.channels, ch), 0, len(self.channels) - 1))
            height = max(float(y[j]) - base, 0.01 * span)
            sigma_ch = float(self.detector.sigma_ev(ref.energy_ev)) / self.calibration.gain
            # Gaussian peak height -> integrated area, in channel units.
            p0.append(height * sigma_ch * np.sqrt(2.0 * np.pi))
            lo.append(0.0)
            hi.append(np.inf)

        if self.fit_resolution:
            p0 += [self.detector.noise_ev, self.detector.fano]
            lo += [1.0, 0.01]
            hi += [200.0, 1.0]
        if self.fit_gamma:
            p0 += [self.detector.gamma_ev]
            lo += [0.0]
            hi += [100.0]

        return p0, (lo, hi)


# =============================================================================
# Fitting
# =============================================================================

@dataclass
class FitResult:
    """Outcome of a constrained fit to one spectrum."""

    model: XRFModel
    popt: np.ndarray
    perr: np.ndarray
    channels: np.ndarray
    y: np.ndarray
    yerr: Optional[np.ndarray]
    yfit: np.ndarray
    chi2_red: float
    areas: Dict[str, float] = field(default_factory=dict)
    area_errors: Dict[str, float] = field(default_factory=dict)

    @property
    def residual(self) -> np.ndarray:
        return self.y - self.yfit

    @property
    def residual_normalised(self) -> np.ndarray:
        """Residual in units of the data uncertainty, where available."""
        if self.yerr is None:
            return self.residual
        return self.residual / np.where(self.yerr > 0, self.yerr, np.nan)

    def params(self) -> Dict[str, Tuple[float, float]]:
        """``{name: (value, 1-sigma)}`` for every free parameter."""
        return {n: (float(v), float(e))
                for n, v, e in zip(self.model.param_names, self.popt, self.perr)}

    def summary(self) -> str:
        lines = [f"reduced chi2 = {self.chi2_red:.3f}", "", "free parameters:"]
        for name, (v, e) in self.params().items():
            lines.append(f"  {name:<20} {v:14.5g} +/- {e:.3g}")
        lines += ["", "group areas (reference line / whole group, same units as data x channel):"]
        for g in self.model.groups:
            a, ae = self.areas[g.label], self.area_errors[g.label]
            lines.append(f"  {g.label:<10} {a:12.5g} +/- {ae:<11.3g}"
                         f"  total {a * g.total_rel:12.5g}  ({len(g.lines)} lines)")
        return "\n".join(lines)


def fit_average_spectrum(
    y: np.ndarray,
    groups: Sequence[LineGroup],
    calibration: Calibration,
    detector: Detector,
    roi: Tuple[int, int],
    yerr: Optional[np.ndarray] = None,
    background_order: Optional[int] = 1,
    fit_resolution: bool = True,
    fit_gamma: bool = True,
    maxfev: int = 20000,
) -> FitResult:
    """Fit the constrained Voigt model to a single spectrum.

    Parameters
    ----------
    y : (n_channels,) array
        Spectrum over the full MCA range; it is sliced to ``roi`` internally.
    groups
        From :func:`element_line_groups`.
    calibration, detector
        Fixed calibration and the detector-response model.
    roi : (lo, hi)
        Half-open channel range to fit.
    yerr : (n_channels,) array, optional
        1-sigma uncertainty per channel, e.g. the standard error returned by
        :func:`average_spectrum`.  Passed to ``curve_fit`` with
        ``absolute_sigma=True`` so that the reported parameter errors are the
        real ones rather than being rescaled to force reduced chi2 = 1.
    background_order, fit_resolution, fit_gamma
        See :class:`XRFModel`.

    Returns
    -------
    FitResult
    """
    lo, hi = roi
    ch = np.arange(lo, hi, dtype=float)
    yy = np.asarray(y, dtype=float)[lo:hi]

    ee = None
    if yerr is not None:
        ee = np.asarray(yerr, dtype=float)[lo:hi]
        # A zero error would make the fit singular; floor it at the smallest
        # positive value present.
        positive = ee[ee > 0]
        floor = positive.min() if positive.size else 1.0
        ee = np.where(ee > 0, ee, floor)

    model = XRFModel(groups, calibration, detector, ch,
                     background_order=background_order,
                     fit_resolution=fit_resolution, fit_gamma=fit_gamma)
    p0, bounds = model.initial_guess(yy)

    popt, pcov = curve_fit(model, ch, yy, p0=p0, bounds=bounds,
                           sigma=ee, absolute_sigma=ee is not None, maxfev=maxfev)
    perr = np.sqrt(np.diag(pcov))
    yfit = model(ch, *popt)

    dof = max(len(yy) - len(popt), 1)
    if ee is not None:
        chi2 = float(np.sum(((yy - yfit) / ee) ** 2)) / dof
    else:
        chi2 = float(np.sum((yy - yfit) ** 2)) / dof

    i0 = model.n_bkg
    areas = {g.label: float(popt[i0 + i]) for i, g in enumerate(model.groups)}
    aerrs = {g.label: float(perr[i0 + i]) for i, g in enumerate(model.groups)}

    return FitResult(model=model, popt=popt, perr=perr, channels=ch, y=yy, yerr=ee,
                     yfit=yfit, chi2_red=chi2, areas=areas, area_errors=aerrs)


# =============================================================================
# Per-pixel mapping: the linear solve
# =============================================================================

@dataclass
class MapResult:
    """Per-pixel amplitudes from the linear map fit.

    Attributes
    ----------
    group_labels : list of str
        Line-group labels, in column order of ``areas``.
    areas, area_errors : (n_pixels, n_groups) arrays
        Fitted reference-line area per pixel and its 1-sigma from the per-pixel
        covariance.
    background : (n_pixels, n_bkg) array
    chi2_red : (n_pixels,) array
    total_rel : dict
        ``{group_label: sum of relative intensities}``, converting a
        reference-line area into the group's total modelled area.
    shape : tuple
        Map shape (n_slow, n_fast), for reshaping.
    """

    group_labels: List[str]
    areas: np.ndarray
    area_errors: np.ndarray
    background: np.ndarray
    chi2_red: np.ndarray
    total_rel: Dict[str, float]
    shape: Tuple[int, int]

    def area_map(self, label: str, total: bool = False) -> np.ndarray:
        """2D map of one group's area. ``total=True`` scales to the whole series."""
        a = self.areas[:, self.group_labels.index(label)]
        if total:
            a = a * self.total_rel[label]
        return a.reshape(self.shape)

    def error_map(self, label: str, total: bool = False) -> np.ndarray:
        e = self.area_errors[:, self.group_labels.index(label)]
        if total:
            e = e * self.total_rel[label]
        return e.reshape(self.shape)

    def snr(self, label: str) -> np.ndarray:
        """Per-pixel signal-to-noise of one group's area."""
        a, e = self.area_map(label), self.error_map(label)
        with np.errstate(divide="ignore", invalid="ignore"):
            return a / e

    def quality_report(self, quantitative_snr: float = 20.0) -> str:
        """Per-group median S/N, flagging groups too weak to quantify.

        A group whose per-pixel S/N is of order a few can still give a usable
        *map-averaged* area, but its per-pixel map is noise and its absolute
        value should not be read as a concentration.  Soft L lines are the usual
        offenders: they are weak to begin with and heavily attenuated by the
        detector window and any air path, so their apparent deficit is an
        instrumental effect, not chemistry.
        """
        lines = [f"{'group':<8}{'median area':>13}{'median S/N':>12}{'neg. px':>9}   status"]
        for lab in self.group_labels:
            a = self.area_map(lab, total=True)
            sn = np.nanmedian(self.snr(lab))
            neg = 100 * np.nanmean(a < 0)
            if sn >= quantitative_snr:
                status = "quantitative"
            elif sn >= 3.0:
                status = "DETECTED ONLY - per-pixel map is noise, area is a lower bound"
            else:
                status = "NOT DETECTED - do not quantify"
            lines.append(f"{lab:<8}{np.nanmedian(a):>13.4g}{sn:>12.1f}{neg:>8.2f}%   {status}")
        return "\n".join(lines)

    def ratio_map(self, numerator: str, denominator: str,
                  total: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Ratio of two groups\' areas, and its propagated 1-sigma error."""
        a, b = self.area_map(numerator, total), self.area_map(denominator, total)
        da, db = self.error_map(numerator, total), self.error_map(denominator, total)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = a / b
            dr = np.abs(r) * np.sqrt((da / a) ** 2 + (db / b) ** 2)
        return r, dr


def design_matrix(model: "XRFModel", shape_params: Sequence[float]) -> np.ndarray:
    """Columns of the linear model: background terms, then one per line group.

    With peak positions and widths fixed, the spectrum is a *linear* combination
    of these columns, the coefficients being the background polynomial terms and
    the group reference-line areas.  Building this once turns a per-pixel
    non-linear optimisation into a single stacked linear solve.

    Parameters
    ----------
    model : XRFModel
        Supplies the channel grid, the groups and the background order.
    shape_params : sequence
        A full parameter vector (e.g. ``FitResult.popt``); only its resolution
        parameters are read, the amplitudes in it are ignored.

    Returns
    -------
    (n_channels, n_bkg + n_groups) array
    """
    cols = [model._u ** k for k in range(model.n_bkg)]
    probe = list(shape_params)
    i0 = model.n_bkg
    for j in range(len(model.groups)):
        unit = probe.copy()
        for jj in range(len(model.groups)):
            unit[i0 + jj] = 1.0 if jj == j else 0.0
        cols.append(model.group_profile(j, unit))
    return np.column_stack(cols)


def fit_map(
    raw_spectra: np.ndarray,
    scale: np.ndarray,
    model: "XRFModel",
    shape_params: Sequence[float],
    map_shape: Tuple[int, int],
    poisson_weights: bool = True,
    clip_negative: bool = False,
) -> MapResult:
    """Solve the fixed-shape model for every pixel of a map, in one pass.

    The normal equations are assembled and solved for all pixels simultaneously
    (``A^T W A`` is only ``k x k`` per pixel, with ``k`` the number of
    components), so the cost is two tensor contractions rather than any
    iteration.  The per-pixel covariance falls out of the same factorisation,
    which is what gives every map an error map for free.

    Parameters
    ----------
    raw_spectra : (n_pixels, n_channels) array
        **Raw** counts.  Required in raw form because Poisson weights are
        defined on counts; the normalisation is applied internally.
    scale : (n_pixels,) array
        Normalisation divisor from :func:`normalise_spectra`.
    model : XRFModel
        Built for the ROI being fitted.
    shape_params : sequence
        Parameter vector from the average-spectrum fit, supplying the fixed
        resolution parameters.
    map_shape : (n_slow, n_fast)
    poisson_weights : bool
        Weight each channel by ``1 / Var(y)``, ``Var(y) = raw / scale^2``.
        Strongly recommended: without it the strongest channels dominate every
        pixel and the reported errors are meaningless.
    clip_negative : bool
        Clip negative fitted amplitudes to zero.  Off by default, because a
        negative amplitude is diagnostic of a mis-specified model or an
        over-flexible background and silently clipping it hides that.

    Returns
    -------
    MapResult
    """
    lo, hi = int(model.channels[0]), int(model.channels[-1]) + 1
    raw = np.asarray(raw_spectra, dtype=float)[:, lo:hi]
    s = np.asarray(scale, dtype=float)
    n_pix, n_ch = raw.shape

    good = np.isfinite(s) & (s > 0)
    s_safe = np.where(good, s, 1.0)
    y = raw / s_safe[:, None]

    A = design_matrix(model, shape_params)                      # (n_ch, k)
    k = A.shape[1]

    if poisson_weights:
        W = (s_safe[:, None] ** 2) / np.maximum(raw, 1.0)
    else:
        W = np.ones_like(y)

    # Stacked normal equations: (A^T W A) c = A^T W y, one k x k system per pixel.
    AtWA = np.einsum("ck,pc,cl->pkl", A, W, A, optimize=True)
    AtWy = np.einsum("ck,pc,pc->pk", A, W, y, optimize=True)

    # Nudge the diagonal so an all-zero pixel does not make the solve singular;
    # such pixels are flagged NaN below via `good`.
    ridge = 1e-12 * np.trace(AtWA, axis1=1, axis2=2)[:, None, None] * np.eye(k)
    coef = np.linalg.solve(AtWA + ridge, AtWy[..., None])[..., 0]
    cov = np.linalg.inv(AtWA + ridge)
    err = np.sqrt(np.clip(np.einsum("pkk->pk", cov), 0.0, None))

    resid = y - coef @ A.T
    dof = max(n_ch - k, 1)
    chi2 = np.einsum("pc,pc,pc->p", resid, W, resid) / dof

    coef[~good] = np.nan
    err[~good] = np.nan
    chi2[~good] = np.nan
    if clip_negative:
        coef[:, model.n_bkg:] = np.clip(coef[:, model.n_bkg:], 0.0, None)

    return MapResult(
        group_labels=[g.label for g in model.groups],
        areas=coef[:, model.n_bkg:],
        area_errors=err[:, model.n_bkg:],
        background=coef[:, :model.n_bkg],
        chi2_red=chi2,
        total_rel={g.label: g.total_rel for g in model.groups},
        shape=tuple(map_shape),
    )


# =============================================================================
# Diagnostics and plotting
# =============================================================================

def identify_peaks(
    y: np.ndarray,
    calibration: Calibration,
    roi: Tuple[int, int],
    prominence_frac: float = 0.01,
    tol_ev: float = 70.0,
    min_rate: float = 0.08,
    z_range: Tuple[int, int] = (12, 83),
    n_candidates: int = 4,
) -> List[dict]:
    """Find peaks and list tabulated lines that could explain them.

    A scouting tool for choosing the element list — it makes no attempt to be a
    quantitative identification, and a close energy match is not evidence on its
    own.  Candidates are ranked by radiative rate.

    Returns
    -------
    list of dict
        ``{"channel", "energy_ev", "height", "candidates": [str, ...]}``.
    """
    from scipy.signal import find_peaks

    lo, hi = roi
    seg = np.asarray(y, dtype=float)[lo:hi]
    idx, _ = find_peaks(seg, prominence=np.nanmax(seg) * prominence_frac, distance=8)

    out = []
    for i in idx:
        ch = lo + int(i)
        e0 = float(calibration.energy(ch))
        hits = []
        for z in range(z_range[0], z_range[1] + 1):
            for name, macro in _LINE_MACROS.items():
                if not _is_iupac_line(name):
                    continue
                e = _line_energy_ev(z, macro)
                if e is None or abs(e - e0) > tol_ev:
                    continue
                r = _line_strength(z, macro, 20.0, "radrate")
                if r >= min_rate:
                    hits.append((r, f"{xraylib.AtomicNumberToSymbol(z)}-{name[:-5]}"
                                    f" {e:.0f} eV (r={r:.2f}, d={e - e0:+.0f})"))
        hits.sort(reverse=True)
        out.append({"channel": ch, "energy_ev": e0, "height": float(seg[i]),
                    "candidates": [h[1] for h in hits[:n_candidates]]})
    return out


def pixel_parameters(model: "XRFModel", map_result: MapResult,
                     shape_params: Sequence[float], pixel: int) -> np.ndarray:
    """Reassemble a full parameter vector for one pixel of a map fit.

    Takes the background coefficients and group amplitudes from the per-pixel
    linear solve, and the resolution parameters from the average-spectrum fit
    that defined the fixed shape.
    """
    p = np.array(shape_params, dtype=float)
    p[:model.n_bkg] = map_result.background[pixel]
    p[model.n_bkg:model.n_bkg + len(model.groups)] = map_result.areas[pixel]
    return p


def select_pixels(map_result: MapResult, n: int = 10, mode: str = "random",
                  seed: Optional[int] = 0) -> np.ndarray:
    """Choose pixel indices to inspect.

    Parameters
    ----------
    n : int
        How many pixels.
    mode : {"random", "worst", "best", "spread"}
        ``"random"`` samples uniformly from pixels with a finite fit;
        ``"worst"``/``"best"`` take the extremes of per-pixel reduced chi2
        (``"worst"`` is the one to look at when diagnosing);
        ``"spread"`` walks the chi2 distribution in equal quantiles, giving a
        representative range rather than a typical one.
    seed : int or None
        Seed for ``mode="random"``.

    Returns
    -------
    (n,) int array of flat pixel indices.
    """
    chi2 = map_result.chi2_red
    ok = np.flatnonzero(np.isfinite(chi2))
    if ok.size == 0:
        raise ValueError("No pixel has a finite fit.")
    n = min(n, ok.size)
    if mode == "random":
        return np.random.default_rng(seed).choice(ok, size=n, replace=False)
    order = ok[np.argsort(chi2[ok])]
    if mode == "best":
        return order[:n]
    if mode == "worst":
        return order[::-1][:n]
    if mode == "spread":
        return order[np.linspace(0, order.size - 1, n).astype(int)]
    raise ValueError(f"mode must be random/worst/best/spread, got {mode!r}")


def plot_pixel_fits(
    raw_spectra: np.ndarray,
    scale: np.ndarray,
    model: "XRFModel",
    map_result: MapResult,
    shape_params: Sequence[float],
    pixels: Optional[Sequence[int]] = None,
    n: int = 10,
    mode: str = "random",
    seed: Optional[int] = 0,
    ncols: int = 5,
    energy_axis: bool = True,
    logy: bool = True,
    show_components: bool = True,
    figsize_per_panel: Tuple[float, float] = (3.2, 2.6),
):
    """Panel grid of individual pixel spectra with their fitted model.

    Map-averaged residuals and a median chi2 both hide what a single spectrum
    actually looks like.  This is the direct check: at a glance you see whether
    the fixed line ratios hold pixel by pixel, whether the background is
    plausible at the local count level, and where the weak groups disappear into
    the noise.

    Parameters
    ----------
    raw_spectra, scale
        As passed to :func:`fit_map`.
    model, map_result, shape_params
        The fitted model, its per-pixel solution, and the shape parameters.
    pixels : sequence of int, optional
        Explicit flat pixel indices.  Overrides ``n``/``mode``/``seed``.
    n, mode, seed
        Passed to :func:`select_pixels` when ``pixels`` is None.
    ncols : int
        Panels per row.
    energy_axis, logy, show_components
        Plot the x axis in keV rather than channels; log the y axis; shade the
        per-group contributions.
    figsize_per_panel : (w, h)
        Size of one panel in inches.

    Returns
    -------
    (fig, axes, pixels)
    """
    import matplotlib.pyplot as plt

    if pixels is None:
        pixels = select_pixels(map_result, n=n, mode=mode, seed=seed)
    pixels = np.asarray(pixels, dtype=int)

    lo, hi = int(model.channels[0]), int(model.channels[-1]) + 1
    raw = np.asarray(raw_spectra, dtype=float)[:, lo:hi]
    s = np.asarray(scale, dtype=float)
    x = model.calibration.energy(model.channels) / 1000.0 if energy_axis else model.channels
    xlabel = "Energy (keV)" if energy_axis else "MCA channel"

    nrows = int(np.ceil(len(pixels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, sharex=True,
                             figsize=(figsize_per_panel[0] * ncols,
                                      figsize_per_panel[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap("tab10")
    n_fast = map_result.shape[1]

    for ax, pix in zip(axes, pixels):
        y = raw[pix] / s[pix]
        params = pixel_parameters(model, map_result, shape_params, pix)
        bkg = model.background(params)
        total = model(model.channels, *params)

        ax.plot(x, y, color="0.25", lw=0.7, drawstyle="steps-mid", label="data")
        if show_components:
            for j, g in enumerate(model.groups):
                comp = model.group_profile(j, params)
                ax.fill_between(x, bkg, bkg + comp, color=cmap(j % 10), alpha=0.45,
                                lw=0, label=g.label)
        ax.plot(x, bkg, "--", color="0.5", lw=0.8)
        ax.plot(x, total, "r-", lw=1.1, label="fit")

        if logy:
            pos = y[y > 0]
            ax.set_yscale("log")
            if pos.size:
                ax.set_ylim(max(pos.min() * 0.5, 1e-4), y.max() * 3)
        row, col = divmod(int(pix), n_fast)
        ax.set_title(f"({row}, {col})   $\\chi^2_\\nu$ = {map_result.chi2_red[pix]:.2f}",
                     fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes[len(pixels):]:
        ax.set_visible(False)
    for ax in axes[max(0, (nrows - 1) * ncols):len(pixels)]:
        ax.set_xlabel(xlabel, fontsize=8)
    for r in range(nrows):
        if r * ncols < len(pixels):
            axes[r * ncols].set_ylabel("counts / live s", fontsize=8)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 7), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Individual pixel fits — {mode} selection of {len(pixels)} "
                 f"(slow, fast) positions", y=1.0, fontsize=11)
    fig.tight_layout()
    return fig, axes, pixels


def plot_fit(result: FitResult, energy_axis: bool = True, logy: bool = False,
             figsize: Tuple[float, float] = (11.0, 7.0)):
    """Two-panel figure: spectrum with per-group components, and the residual.

    Returns
    -------
    (fig, (ax_spectrum, ax_residual))
    """
    import matplotlib.pyplot as plt

    model = result.model
    x = model.calibration.energy(result.channels) / 1000.0 if energy_axis else result.channels
    xlabel = "Energy (keV)" if energy_axis else "MCA channel"

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.07})

    if result.yerr is not None:
        ax.fill_between(x, result.y - result.yerr, result.y + result.yerr,
                        color="0.75", lw=0, label=r"data $\pm 1\sigma$")
    ax.plot(x, result.y, "k-", lw=1.0, label="map-averaged data")

    bkg = model.background(result.popt)
    ax.plot(x, bkg, "--", color="0.45", lw=1.0, label="background")

    cmap = plt.get_cmap("tab10")
    for i, g in enumerate(model.groups):
        comp = model.group_profile(i, result.popt)
        ax.fill_between(x, bkg, bkg + comp, color=cmap(i % 10), alpha=0.45, lw=0,
                        label=f"{g.label}  ({len(g.lines)} lines)")
        # mark the group's reference line
        xref = (g.reference.energy_ev / 1000.0 if energy_axis
                else model.calibration.channel(g.reference.energy_ev))
        ax.axvline(xref, color=cmap(i % 10), lw=0.8, ls=":", alpha=0.8)

    ax.plot(x, result.yfit, "r-", lw=1.4, label="total fit")

    ax.set_ylabel("Counts per live second (monitor-normalised)")
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(max(result.y[result.y > 0].min() * 0.5, 1e-3), result.y.max() * 2)
    ax.legend(ncol=2, fontsize=9, framealpha=0.9)
    ax.set_title(f"Constrained Voigt fit — reduced $\\chi^2$ = {result.chi2_red:.2f}")

    r = result.residual_normalised if result.yerr is not None else result.residual
    axr.axhline(0.0, color="0.5", lw=0.8)
    axr.plot(x, r, "k-", lw=0.8)
    axr.set_xlabel(xlabel)
    axr.set_ylabel(r"residual / $\sigma$" if result.yerr is not None else "residual")

    fig.align_ylabels()
    return fig, (ax, axr)
