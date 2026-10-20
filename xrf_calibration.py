"""
Module: xrf_calibration.py

Energy calibration (channel -> energy) of an energy-dispersive detector from
the spectrum of a reference sample.

    from xrf_calibration import CALIBRANTS, calibrate, plot_calibration

    result = calibrate(spectrum, CALIBRANTS["AXO RF C10"], detector,
                       extra_elements=["Ar"], incident_spectrum=(E_kev, phi))
    print(result.summary())
    CALIB = result.calibration          # xrf_fit.Calibration, for the analysis

How it works
------------
1. **Coarse match.**  The tabulated lines of the calibrant are compared with
   the measured peaks over a grid of gains and offsets around the starting
   guess; the pair that puts the most line intensity on top of peaks wins.
   No peak has to be picked or assigned by hand.
2. **Global fit.**  The constrained model of :mod:`xrf_fit` is fitted with the
   gain and offset free (``fit_calibration=True``): every line of every
   element sits at its tabulated energy, with tabulated relative intensities,
   so unresolved multiplets (K-alpha1/2, L-beta1-4, ...) are placed correctly
   instead of being treated as single peaks.  The detector resolution is
   fitted at the same time, and optionally the sample-detector air path
   (``fit_air_path=True``), from the intensity ratios *within* each element
   (K-alpha/K-beta, L-alpha/L-beta/L-gamma).  Those ratios do not depend on
   the loadings; the L ratios depend somewhat on the excitation spectrum,
   which is the main systematic error of the fitted path.
3. **Line checks.**  The strongest peaks are refitted one at a time in a
   narrow window, with only an energy shift free.  The shifts should scatter
   around zero within their errors; a trend with energy reveals a
   non-linear response that a gain and offset cannot describe.
4. **Intensity check.**  For a thin-film calibrant with certified loadings,
   the reference-line area of each element should be ``A = k * loading * S``,
   with ``S`` the group's ``ref_strength`` (excitation x cross-section x
   detection efficiency, at the fitted air path) and ``k`` one overall scale.
   Comparing the two tests the excitation and efficiency models; it does not
   affect the calibration.

Calibrants
----------
A :class:`Calibrant` lists the certified elements and their loadings.  To add
a new one, add an entry to :data:`CALIBRANTS`.  Thin-film standards (a few
atomic layers on a membrane, like the AXO samples) need no matrix correction,
which is what the intensity check assumes.

Authors: Samuel Tardif, with Claude (Anthropic).
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import xraylib
from scipy.ndimage import gaussian_filter1d

from scipy.optimize import curve_fit

from xrf_beamline import AIR_FORMULA, air_density
from xrf_fit import (Calibration, Detector, FitResult, LineGroup, element_line_groups,
                     fit_average_spectrum, plot_fit, snip_background)

logger = logging.getLogger(__name__)

#: Siegbahn names of the common lines, for labels (IUPAC name -> Siegbahn).
SIEGBAHN = {"KL3": "Ka1", "KL2": "Ka2", "KM3": "Kb1", "KM2": "Kb3", "KN3": "Kb2",
            "L3M5": "La1", "L3M4": "La2", "L2M4": "Lb1", "L3N5": "Lb2", "L1M3": "Lb3",
            "L1M2": "Lb4", "L2N4": "Lg1", "L1N3": "Lg3", "L3M1": "Ll", "L2M1": "Ln",
            "M5N7": "Ma1", "M4N6": "Mb"}

__all__ = ["Calibrant", "CALIBRANTS", "LineCheck", "CalibrationResult",
           "coarse_calibration", "calibrate", "plot_calibration", "calibration_from_peaks"]


# =============================================================================
# Reference samples
# =============================================================================

@dataclass(frozen=True)
class Calibrant:
    """A reference sample.

    Parameters
    ----------
    name : str
    loadings_ng_mm2 : mapping
        Certified areal density of each element, ng/mm^2.  Only the ratios
        matter here; the elements' lines are what the calibration uses.
    description : str
        Free text (supplier, reference, substrate).
    """

    name: str
    loadings_ng_mm2: Mapping[str, float]
    description: str = ""

    @property
    def elements(self) -> List[str]:
        return list(self.loadings_ng_mm2)

    def atom_ratio(self, element: str, reference: str) -> float:
        """Certified atom-number ratio ``n_element / n_reference``."""
        def atoms(el: str) -> float:
            return self.loadings_ng_mm2[el] / xraylib.AtomicWeight(xraylib.SymbolToAtomicNumber(el))
        return atoms(element) / atoms(reference)


#: Known reference samples, by name.
CALIBRANTS: Dict[str, Calibrant] = {
    "AXO RF C10": Calibrant(
        name="AXO RF C10",
        loadings_ng_mm2={"Pb": 84.9, "La": 121.4, "Pd": 23.3, "Mo": 8.6, "Cu": 22.2, "Fe": 43.9},
        description=("AXO Dresden RF-type XRF reference, C10 loading (RF-200-0205-C10-X): "
                     "a few atomic layers of each element on a 200 nm Si membrane. "
                     "https://www.axo-dresden.de/en/products/reference-samples/"),
    ),
}


# =============================================================================
# Results
# =============================================================================

@dataclass
class LineCheck:
    """Position of one peak measured on its own, relative to the calibration.

    ``shift_ev`` = energy that the calibration gives the measured peak minus
    the tabulated line energy.  > 0: the peak sits at a higher channel than
    the calibration predicts.
    """

    label: str            # e.g. "Pb LA1"
    energy_ev: float      # tabulated energy of the line
    channel: float        # where the calibration puts it
    shift_ev: float
    shift_err_ev: float


@dataclass
class CalibrationResult:
    """Outcome of :func:`calibrate`.

    Attributes
    ----------
    calibration : Calibration
        The fitted ``E = gain * channel + offset``.
    covariance : (2, 2) array
        Covariance of (gain, offset).
    start : Calibration
        Result of the coarse match (the starting point of the fit).
    fit : FitResult
        The global fit (``plot_fit(result.fit)`` shows it).
    detector : Detector
        The input detector with the fitted resolution (noise, Fano, gamma)
        and, if fitted, air path.
    checks : list of LineCheck
        Individual peak positions (empty if not requested).
    intensity : dict
        Measured vs predicted areas of the certified elements: ``rows``,
        ``scale``, ``chi2_red``, ``air_path_cm``.  Empty without enough
        certified elements in the spectrum.
    air_path_err : float
        1-sigma statistical error of the fitted air path (NaN if not fitted).
    """

    calibration: Calibration
    covariance: np.ndarray
    start: Calibration
    fit: FitResult
    detector: Detector
    calibrant: Calibrant
    checks: List[LineCheck] = field(default_factory=list)
    intensity: dict = field(default_factory=dict)
    air_path_err: float = float("nan")

    @property
    def gain_err(self) -> float:
        return float(np.sqrt(self.covariance[0, 0]))

    @property
    def offset_err(self) -> float:
        return float(np.sqrt(self.covariance[1, 1]))

    def energy_error(self, channel: np.ndarray | float) -> np.ndarray | float:
        """1-sigma uncertainty of the calibrated energy at a channel (eV),
        including the gain-offset correlation."""
        c = np.asarray(channel, dtype=float)
        cov = self.covariance
        return np.sqrt(c ** 2 * cov[0, 0] + 2 * c * cov[0, 1] + cov[1, 1])

    def check_chi2(self) -> float:
        """Reduced chi2 of the line shifts against zero: ~1 when the response
        is linear within the measured precision."""
        if not self.checks:
            return float("nan")
        z = np.array([c.shift_ev / c.shift_err_ev for c in self.checks])
        return float(np.mean(z ** 2))

    def summary(self) -> str:
        c, d = self.calibration, self.detector
        lines = [
            f"Calibrant: {self.calibrant.name}",
            f"E[eV] = {c.gain:.5f} (+/- {self.gain_err:.5f}) * channel "
            f"{c.offset:+.2f} (+/- {self.offset_err:.2f})",
            f"  gain-offset correlation {self.covariance[0, 1] / (self.gain_err * self.offset_err):+.3f}; "
            f"energy uncertainty {self.energy_error(c.channel(5000)):.2f} eV at 5 keV, "
            f"{self.energy_error(c.channel(15000)):.2f} eV at 15 keV",
            f"  coarse match was gain {self.start.gain:.4f}, offset {self.start.offset:+.1f} eV",
            f"global fit: reduced chi2 = {self.fit.chi2_red:.2f}",
            f"resolution: FWHM {d.fwhm_ev(5900):.0f} eV at 5.9 keV, {d.fwhm_ev(10000):.0f} eV at 10 keV "
            f"(noise {d.noise_ev:.1f} eV, Fano {d.fano:.3f}, gamma {d.gamma_ev:.1f} eV)",
            (f"sample-detector air path: {d.air_path_cm:.2f} +/- {self.air_path_err:.2f} cm (fitted, "
             f"statistical error only)" if np.isfinite(self.air_path_err)
             else f"sample-detector air path: {d.air_path_cm:.2f} cm (fixed)"),
        ]
        if self.checks:
            lines.append(f"line checks (calibrated energy of the peak - tabulated energy), "
                         f"chi2 vs 0 = {self.check_chi2():.2f}:")
            for k in sorted(self.checks, key=lambda k: k.energy_ev):
                lines.append(f"  {k.label:<16} {k.energy_ev / 1000:7.3f} keV  ch {k.channel:7.1f}  "
                             f"{k.shift_ev:+6.1f} +/- {k.shift_err_ev:4.1f} eV")
        if self.intensity:
            it = self.intensity
            lines.append(f"intensity check, area = k x loading x S(E) at {it['air_path_cm']:.1f} cm of air "
                         f"(tests the excitation/efficiency model), chi2 = {it['chi2_red']:.1f}:")
            for r in it["rows"]:
                note = "" if r["used"] else "   (below 3 sigma, not used)"
                lines.append(f"  {r['label']:<6} {r['energy_ev'] / 1000:6.2f} keV  measured / predicted "
                             f"{r['ratio']:5.2f} +/- {r['ratio_err']:.2f}{note}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Plain-data record, e.g. to save as JSON next to the data."""
        c, d = self.calibration, self.detector
        return {
            "calibrant": self.calibrant.name,
            "gain_ev_per_channel": c.gain, "gain_err": self.gain_err,
            "offset_ev": c.offset, "offset_err": self.offset_err,
            "covariance_gain_offset": self.covariance.tolist(),
            "chi2_red": self.fit.chi2_red,
            "detector": {"noise_ev": d.noise_ev, "fano": d.fano, "gamma_ev": d.gamma_ev,
                         "fwhm_ev_at_5900": float(d.fwhm_ev(5900)), "air_path_cm": d.air_path_cm,
                         "air_path_err": self.air_path_err},
            "line_checks": [dataclasses.asdict(k) for k in self.checks],
            "intensity": self.intensity,
        }


# =============================================================================
# Procedure
# =============================================================================

def coarse_calibration(
    y: np.ndarray,
    groups: Sequence[LineGroup],
    gain: float = 10.0,
    offset: float = 0.0,
    gain_tolerance: float = 0.1,
    offset_tolerance_ev: float = 500.0,
    smooth_ch: float = 4.0,
) -> Calibration:
    """Best (gain, offset) on a grid, by matching the line pattern.

    The background (SNIP) is removed, the spectrum smoothed and square-rooted
    (so that weak elements count too), and each candidate calibration is
    scored by the sum of that signal at the predicted channels of all lines,
    weighted by their relative intensities.

    Parameters
    ----------
    y : (n_channels,) array
    groups : line groups of the calibrant (:func:`xrf_fit.element_line_groups`)
    gain, offset : starting guess (eV/channel, eV)
    gain_tolerance : searched gain range, as a fraction of ``gain``
    offset_tolerance_ev : searched offset range, +/- eV
    smooth_ch : Gaussian smoothing of the spectrum, channels
    """
    y = np.asarray(y, dtype=float)
    signal = np.sqrt(gaussian_filter1d(np.clip(y - snip_background(y, width=40), 0, None), smooth_ch))
    e = np.array([l.energy_ev for g in groups for l in g.lines])
    w = np.array([l.rel for g in groups for l in g.lines])

    gains = gain * np.linspace(1 - gain_tolerance, 1 + gain_tolerance, 801)
    offsets = offset + np.arange(-offset_tolerance_ev, offset_tolerance_ev + 1e-9, 2.0)
    best = (-np.inf, gain, offset)
    for g in gains:                              # one gain at a time keeps memory small
        ch = (e[None, :] - offsets[:, None]) / g
        score = (np.interp(ch, np.arange(y.size), signal, left=0.0, right=0.0) * w).sum(axis=1)
        j = int(np.argmax(score))
        if score[j] > best[0]:
            best = (float(score[j]), float(g), float(offsets[j]))
    _, g_best, o_best = best
    if abs(g_best / gain - 1) > 0.99 * gain_tolerance or abs(o_best - offset) > 0.99 * offset_tolerance_ev:
        logger.warning("Coarse calibration at the edge of the search range (gain %.4f, offset %.0f eV): "
                       "widen it or improve the starting guess.", g_best, o_best)
    return Calibration(gain=g_best, offset=o_best)


def calibration_from_peaks(y: np.ndarray, peaks: Sequence[Tuple[float, float]],
                           half_width: int = 35) -> Tuple[Calibration, List[dict]]:
    """Linear calibration through reference peaks found near given channels.

    Each peak is located by a Gaussian plus linear-background fit within
    ``half_width`` channels of its starting channel; the calibration is the
    least-squares line (exact with two peaks) through (centroid, energy).

    Parameters
    ----------
    y : (n_channels,) array
    peaks : sequence of (channel guess, energy in eV)
        For an unresolved doublet give its intensity-weighted energy (e.g.
        K-alpha1/K-alpha2), since that is where the centroid sits.

    Returns
    -------
    (Calibration, list of {"channel", "channel_err", "sigma_ch", "energy_ev"})
    """
    y = np.asarray(y, dtype=float)

    def model(x, a, c, s, b0, b1):
        return a * np.exp(-0.5 * ((x - c) / s) ** 2) + b0 + b1 * (x - c)

    found = []
    for c0, e in peaks:
        lo, hi = max(int(c0) - half_width, 0), min(int(c0) + half_width + 1, y.size)
        x, yy = np.arange(lo, hi, dtype=float), y[lo:hi]
        p0 = [float(yy.max() - yy.min()), float(x[np.argmax(yy)]), half_width / 4.0, float(yy.min()), 0.0]
        popt, pcov = curve_fit(model, x, yy, p0=p0, sigma=np.sqrt(np.maximum(yy, 1.0)), maxfev=5000)
        found.append({"channel": float(popt[1]), "channel_err": float(np.sqrt(pcov[1, 1])),
                      "sigma_ch": abs(float(popt[2])), "energy_ev": float(e)})
    ch = np.array([f["channel"] for f in found])
    en = np.array([f["energy_ev"] for f in found])
    gain, offset = np.polyfit(ch, en, 1)
    return Calibration(gain=float(gain), offset=float(offset)), found


def _line_checks(y: np.ndarray, yerr: Optional[np.ndarray], fit: FitResult,
                 detector: Detector, escape: bool,
                 refit_groups: Optional[Sequence[LineGroup]] = None,
                 min_fraction: float = 0.05, half_width_fwhm: float = 1.5) -> List[LineCheck]:
    """Refit the strongest peaks one at a time with only an energy shift free.

    Each window keeps the groups that contribute at least 2 % of the modelled
    peak signal there (from the global fit); weaker ones would only add
    amplitudes the window cannot constrain.  ``refit_groups`` (matched by
    label) replace the global fit's groups in the refits, e.g. groups rebuilt
    at a fitted air path.
    """
    cal, groups, n = fit.calibration, fit.model.groups, len(y)
    replace = {g.label: g for g in (refit_groups or [])}
    profiles = [fit.model.group_profile(i, fit.popt) for i in range(len(groups))]
    first = int(fit.channels[0])
    # candidate peaks: fitted area of each line, strongest first
    peaks = sorted(((fit.areas[g.label] * l.rel, g, l) for g in groups for l in g.lines),
                   key=lambda t: -t[0])
    top = peaks[0][0]
    chosen: List[Tuple[LineGroup, object]] = []
    for area, g, l in peaks:
        if area < min_fraction * top:
            break
        fwhm = detector.fwhm_ev(l.energy_ev)
        if all(abs(l.energy_ev - m.energy_ev) > fwhm for _, m in chosen):
            chosen.append((g, l))

    checks = []
    for g, line in chosen:
        half = half_width_fwhm * detector.fwhm_ev(line.energy_ev)
        roi = cal.channel_range(line.energy_ev - half, line.energy_ev + half, n)
        if roi[1] - roi[0] < 8:
            continue
        a, b = max(roi[0] - first, 0), max(roi[1] - first, 0)
        peak = [prof[a:b].max() if b > a else 0.0 for prof in profiles]
        local = [replace.get(gg.label, gg) for gg, h in zip(groups, peak) if h >= 0.02 * max(peak)]
        try:
            r = fit_average_spectrum(y, local, cal, detector, roi, yerr=yerr, background_order=1,
                                     fit_resolution=False, fit_gamma=False, escape=escape,
                                     fit_calibration="offset", maxfev=5000)
        except (RuntimeError, ValueError) as err:
            logger.warning("Line check %s %s failed: %s", g.element, line.name, err)
            continue
        value, error = r.params()["cal_offset_ev"]
        # a positive shift of the offset moves predicted peaks up; the peak's
        # measured energy relative to the calibration is minus that
        name = f"{SIEGBAHN[line.name]} ({line.name})" if line.name in SIEGBAHN else line.name
        checks.append(LineCheck(label=f"{g.element} {name}", energy_ev=line.energy_ev,
                                channel=float(cal.channel(line.energy_ev)),
                                shift_ev=-(value - cal.offset), shift_err_ev=error))
    return checks


def _intensity_model(fit: FitResult, calibrant: Calibrant, detector: Detector,
                     air_path_cm: float) -> dict:
    """Compare the reference-line areas of the certified elements with
    ``A = k * loading * S(E_ref)``, one family per element (the most precise).

    ``S`` is the group's ``ref_strength``, computed with
    ``detector.air_path_cm = d0``; at the air path ``air_path_cm = d`` it
    scales as ``exp(-mu_air(E_ref) (d - d0))``.  Only ``k`` is fitted, on the
    elements measured above 3 sigma.
    """
    best: Dict[str, LineGroup] = {}
    snr = {g.label: fit.areas[g.label] / max(fit.area_errors[g.label], 1e-300) for g in fit.model.groups}
    for g in fit.model.groups:
        if g.element in calibrant.loadings_ng_mm2 and np.isfinite(g.ref_strength) and g.ref_strength > 0:
            if g.element not in best or snr[g.label] > snr[best[g.element].label]:
                best[g.element] = g
    groups = [best[el] for el in calibrant.elements if el in best]
    area = np.array([fit.areas[g.label] for g in groups])
    err = np.array([fit.area_errors[g.label] for g in groups])
    used = area > 3 * err
    if used.sum() < 2:
        return {}

    rho = air_density(detector.air_temperature_c, detector.air_pressure_hpa)
    mu = np.array([xraylib.CS_Total_CP(AIR_FORMULA, g.reference.energy_ev / 1000.0) * rho for g in groups])
    m = np.array([calibrant.loadings_ng_mm2[g.element] * g.ref_strength for g in groups])
    m *= np.exp(-mu * (air_path_cm - detector.air_path_cm))
    w = 1.0 / err[used] ** 2
    k = float(np.sum(w * area[used] * m[used]) / np.sum(w * m[used] ** 2))
    pred = k * m
    chi2 = float(np.sum(((area[used] - pred[used]) / err[used]) ** 2) / max(int(used.sum()) - 1, 1))
    return {"air_path_cm": float(air_path_cm), "scale": k, "chi2_red": chi2, "rows": [
        {"label": g.label, "energy_ev": g.reference.energy_ev,
         "loading_ng_mm2": calibrant.loadings_ng_mm2[g.element],
         "area": float(a), "area_err": float(e), "predicted": float(pr),
         "ratio": float(a / pr), "ratio_err": float(e / pr), "used": bool(u)}
        for g, a, e, pr, u in zip(groups, area, err, pred, used)]}


def calibrate(
    y: np.ndarray,
    calibrant: Calibrant,
    detector: Detector,
    yerr: Optional[np.ndarray] = None,
    roi_ev: Tuple[float, float] = (2500.0, 19500.0),
    gain: float = 10.0,
    offset: float = 0.0,
    extra_elements: Sequence[str] = (),
    families: Sequence[str] = ("K", "L", "M"),
    incident_spectrum: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    e0_eff_kev: float = 20.0,
    background_order: int = 5,
    escape: bool = True,
    min_rel: float = 0.01,
    coarse: bool = True,
    check_lines: bool = True,
    fit_air_path: bool = False,
    **coarse_kw,
) -> CalibrationResult:
    """Calibrate the channel -> energy relation on a reference spectrum.

    Parameters
    ----------
    y : (n_channels,) array
        Spectrum of the calibrant (raw counts, or normalised).  A single
        measurement or a sum / average of several.
    calibrant : Calibrant
        For instance ``CALIBRANTS["AXO RF C10"]``.
    detector : Detector
        Starting resolution and the detection-efficiency model.
    yerr : (n_channels,) array, optional
        1-sigma errors; default Poisson, ``sqrt(max(y, 1))`` (only right for
        raw counts).
    roi_ev : (min, max)
        Energy window fitted, in eV.
    gain, offset : float
        Starting guess, eV/channel and eV.
    extra_elements : sequence of str
        Elements present besides the certified ones: argon from the air path,
        contamination...  They get their own free amplitudes.
    families : shells considered for every element (only lines inside the ROI are kept)
    incident_spectrum, e0_eff_kev
        Excitation, as in :func:`xrf_fit.element_line_groups`.  It sets the
        relative line intensities within each group (and the intensity check).
    background_order : polynomial order of the background over the ROI
    escape : model the sensor escape peaks
    min_rel : drop lines weaker than this fraction of their group's reference
    coarse : run the coarse match first (else start from ``gain``, ``offset``)
    check_lines : measure the strongest peaks individually
    fit_air_path : fit the sample-detector air path (from the line ratios
        within each element), starting from ``detector.air_path_cm``
    **coarse_kw : passed to :func:`coarse_calibration` (search ranges)
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    yerr = np.sqrt(np.maximum(y, 1.0)) if yerr is None else np.asarray(yerr, dtype=float).reshape(-1)
    elements = list(dict.fromkeys(list(calibrant.elements) + list(extra_elements)))
    groups = element_line_groups(elements, *roi_ev, e0_eff_kev=e0_eff_kev, families=families,
                                 min_rel=min_rel, incident_spectrum=incident_spectrum,
                                 detector=detector)
    if not groups:
        raise ValueError(f"No line of {elements} between {roi_ev[0]:.0f} and {roi_ev[1]:.0f} eV.")

    start = (coarse_calibration(y, groups, gain, offset, **coarse_kw) if coarse
             else Calibration(gain=gain, offset=offset))
    roi = start.channel_range(*roi_ev, y.size)
    fit = fit_average_spectrum(y, groups, start, detector, roi, yerr=yerr,
                               background_order=background_order, escape=escape,
                               fit_calibration=True, fit_air_path=fit_air_path)
    p = fit.params()
    i = fit.model.param_names.index("cal_gain")
    covariance = np.array(fit.pcov[i:i + 2, i:i + 2], dtype=float)
    air_path = fit.model.air_path_of(fit.popt)
    fitted_detector = dataclasses.replace(detector, noise_ev=p["noise_ev"][0], fano=p["fano"][0],
                                          gamma_ev=p["gamma_ev"][0], air_path_cm=air_path)

    result = CalibrationResult(calibration=fit.calibration, covariance=covariance, start=start,
                               fit=fit, detector=fitted_detector, calibrant=calibrant,
                               intensity=_intensity_model(fit, calibrant, detector, air_path),
                               air_path_err=p["air_path_cm"][1] if fit_air_path else float("nan"))
    if check_lines:
        # local refits use the fitted response, with the air path fixed: rebuild
        # the groups at that path so their relative intensities include it
        local_groups = (element_line_groups(elements, *roi_ev, e0_eff_kev=e0_eff_kev, families=families,
                                            min_rel=min_rel, incident_spectrum=incident_spectrum,
                                            detector=fitted_detector) if fit_air_path else groups)
        result.checks = _line_checks(y, yerr, fit, fitted_detector, escape, refit_groups=local_groups)
    return result


def plot_calibration(result: CalibrationResult, logy: bool = True):
    """Global fit (spectrum and residual) and the line checks.

    Returns
    -------
    (fig_fit, fig_checks) ; ``fig_checks`` is None without line checks.
    """
    import matplotlib.pyplot as plt

    fig_fit, (ax, _) = plot_fit(result.fit, energy_axis=True, logy=logy, data_label="data",
                                ylabel="Intensity (input units)")
    c = result.calibration
    ax.set_title(f"{result.calibrant.name}: E = {c.gain:.4f} x ch {c.offset:+.1f} eV, "
                 f"reduced $\\chi^2$ = {result.fit.chi2_red:.2f}")
    if not result.checks:
        return fig_fit, None

    fig, ax = plt.subplots(figsize=(8, 4))
    k = sorted(result.checks, key=lambda k: k.energy_ev)
    e = np.array([x.energy_ev for x in k]) / 1000
    ax.errorbar(e, [x.shift_ev for x in k], yerr=[x.shift_err_ev for x in k], fmt="o", capsize=3)
    for x, ei in zip(k, e):
        ax.annotate(x.label.split(" (")[0], (ei, x.shift_ev), textcoords="offset points",
                    xytext=(4, 6), fontsize=8)
    grid = np.linspace(min(e.min(), 1.0), max(e.max(), 20.0), 200)
    band = result.energy_error(c.channel(grid * 1000))
    ax.fill_between(grid, -band, band, color="C0", alpha=0.15, lw=0, label="calibration 1$\\sigma$")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlabel("Line energy (keV)")
    ax.set_ylabel("Calibrated peak energy - tabulated (eV)")
    ax.set_title(f"Individual peak positions ($\\chi^2$ vs 0 = {result.check_chi2():.2f})")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig_fit, fig
