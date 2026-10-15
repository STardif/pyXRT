"""
Module: xrf_beamline.py

Incident white-beam spectrum of a bending-magnet beamline (BM32, ESRF).

Computes the Schwinger spectral flux of the source and propagates it through a
list of optical elements (grazing-incidence mirrors, transmission windows and
filters) to give Phi(E) at the sample.  That spectrum is what weights the
fluorescence cross-sections in :mod:`xrf_fit` when the illumination is
polychromatic.

Why this matters for the line ratios
------------------------------------
With a monochromatic beam every subshell of an element sees the same photon
energy, and the relative intensity of its lines is a property of the element
alone.  Under white-beam illumination each subshell is excited by the part of
the spectrum lying above *its own* edge, and those edges can be far apart
compared with the width of the band:

=======  ==========================  ======================================
element  L3 / L2 / L1 edge (keV)     fraction of a 5-28 keV band above each
=======  ==========================  ======================================
Nd       6.21 / 6.72 / 7.13          nearly all of it, for every subshell
Ir       11.22 / 12.82 / 13.42       noticeably different per subshell
Pb       13.04 / 15.20 / 15.86       strongly different per subshell
=======  ==========================  ======================================

So the heavier the element, the more a single effective energy misrepresents
the Lb/La and Lg/La ratios.  Flux weighting removes that error.

Conventions and caveats
-----------------------
* ``spectral_flux`` returns photons/s/mrad(horizontal)/0.1 % bandwidth
  integrated over the full vertical opening angle,
  ``2.457e13 * E[GeV] * I[A] * G1(y)``; ``angular_flux`` returns the angular
  density photons/s/mrad^2/0.1 % BW, ``1.327e13 * E[GeV]^2 * I[A] * F(y, psi)``,
  whose on-axis value is the familiar ``H2(y)`` form.  **The current is in
  amperes.**  (Before Oct 2026 ``spectral_flux`` multiplied ``G1`` by the
  angular-density constant ``1.327e13 E^2 I``, a constant factor E/1.85 too
  high; irrelevant for ratios, wrong for absolute flux.)
* **Vertical acceptance.**  The photon beam of a bending magnet is
  ~1/gamma = 85 urad wide vertically at ESRF-EBS (FWHM 110-270 urad between 30
  and 5 keV), and softer photons spread wider.  When the beamline apertures
  accept much less than that, as on BM32 where slit s3 passes a few urad, the
  accepted spectrum is the *on-axis* spectrum, noticeably harder than the
  vertically integrated ``G1`` shape.  :class:`Beamline` therefore traces the
  apertures (:class:`Slit`, mirror footprints) back to the source, through the
  bent mirrors, and integrates :meth:`BendingMagnetSource.angular_flux` over
  the accepted vertical window.  On the 2026 BM32 maps this changes the Ga/As
  sensitivity ratio by ~4.5 %.  Horizontal apertures only scale the flux (the
  fan is uniform in the horizontal angle) and do not change the spectrum.
* **KB mirrors.**  Along an elliptical focusing mirror the grazing angle
  varies, because rays from a distant source arrive almost parallel but leave
  converging on the focus: over an illuminated beam of size ``s`` at image
  distance ``q`` the angle spans ``theta0 +/- s/(4q)``.  This smooths the
  mirror's high-energy cut-off and is applied to mirrors given an
  ``image_distance_m``.
* **Per-bandwidth vs per-energy.**  A 0.1 % bandwidth is ``dE = 1e-3 * E``, so
  the flux per unit energy is ``dN/dE = N_0.1%BW / (1e-3 * E)``.  Any integral
  over ``dE`` (e.g. a flux-weighted cross-section ``int sigma(E) Phi(E) dE``)
  must use the per-energy density, otherwise high energies are over-weighted
  by a factor E.  Use :func:`per_unit_energy` (or
  :meth:`Beamline.spectral_density`) for the conversion; ``xrf_fit`` applies
  it internally to the per-0.1 %BW spectrum returned by
  :meth:`Beamline.spectrum`.
* Mirror reflectivity is s-polarised Fresnel with an optional Nevot-Croce
  roughness factor.  Synchrotron radiation is predominantly horizontally
  polarised and the mirrors here deflect vertically, so s-polarisation is the
  relevant case; at grazing incidence the two are indistinguishable anyway.
* The default BM32 configuration (0.86 T bending magnet, four Ir-coated
  mirrors, two 500 um Be windows, 3.3 m measured incident air path, element
  positions in :data:`BM32_LAYOUT`) was confirmed in Oct 2026.  Mirror
  roughness is not known and is left at 0; the KB grazing angles are not
  recorded in the scan files and default to 3 mrad.  Use :func:`bm32_from_h5`
  to build the beamline with the slit gaps, mirror angles, bending radii and
  ring current of a given scan.

Authors: Samuel Tardif, with Claude (Anthropic).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import xraylib
from scipy.integrate import quad, trapezoid
from scipy.special import kv

logger = logging.getLogger(__name__)
xraylib.SetErrorMessages(0)

__all__ = [
    "BendingMagnetSource",
    "Mirror",
    "Filter",
    "Slit",
    "Beamline",
    "BM32_LAYOUT",
    "BM32_WHITE_BEAM",
    "bm32_from_h5",
    "air_path",
    "air_density",
    "AIR_FORMULA",
    "per_unit_energy",
]


# =============================================================================
# Source
# =============================================================================

@dataclass
class BendingMagnetSource:
    """Synchrotron radiation of a bending-magnet (or short wiggler) source.

    Parameters
    ----------
    ring_energy_gev : float
        Storage-ring energy.  6.04 GeV for ESRF-EBS.
    magnetic_field_t : float
        Dipole field.  Sets the critical energy, to which the high-energy
        spectrum is exponentially sensitive — confirm it for your source
        (an EBS two-pole wiggler is not the same as a standard EBS dipole).
        Default 0.86 T: the BM32 bending magnet (confirmed Oct 2026).
    current_a : float
        Ring current in **amperes** (0.2 A = 200 mA).
    """

    ring_energy_gev: float = 6.04
    magnetic_field_t: float = 0.86
    current_a: float = 0.2

    @property
    def critical_energy_kev(self) -> float:
        """``Ec = 0.665 * E[GeV]^2 * B[T]`` in keV."""
        return 0.665 * self.ring_energy_gev ** 2 * self.magnetic_field_t

    @property
    def gamma(self) -> float:
        """Lorentz factor of the electrons."""
        return self.ring_energy_gev * 1e3 / 0.51099895

    @staticmethod
    def _g1(y: float) -> float:
        """``G1(y) = y * integral_y^inf K_{5/3}(x) dx`` — vertically integrated shape."""
        if y <= 0:
            return 0.0
        if y > 50.0:                      # underflows; the tail is negligible here
            return 0.0
        val, _ = quad(lambda x: kv(5.0 / 3.0, x), y, np.inf, epsabs=1e-12, limit=200)
        return y * val

    def spectral_flux(self, energies_kev: np.ndarray) -> np.ndarray:
        """Flux in photons/s/mrad(horizontal)/0.1 % bandwidth, integrated over
        the full vertical opening angle: ``2.457e13 E[GeV] I[A] G1(E/Ec)``."""
        e = np.atleast_1d(np.asarray(energies_kev, dtype=float))
        y = e / self.critical_energy_kev
        shape = np.array([self._g1(float(yy)) for yy in y])
        return 2.457e13 * self.ring_energy_gev * self.current_a * shape

    def angular_flux(self, energies_kev: np.ndarray, psi_urad: np.ndarray) -> np.ndarray:
        """Angular density, photons/s/mrad^2/0.1 % BW, both polarisations.

        ``1.327e13 E^2 I y^2 (1+X^2)^2 [K_{2/3}^2(xi) + X^2/(1+X^2) K_{1/3}^2(xi)]``
        with ``y = E/Ec``, ``X = gamma psi`` and ``xi = (y/2)(1+X^2)^{3/2}``; the
        first term is the horizontally (sigma) polarised part.  On axis it
        reduces to ``1.327e13 E^2 I H2(y)``; integrated over ``psi`` (in mrad)
        it gives :meth:`spectral_flux`.

        Returns an array of shape ``(n_energies, n_angles)``.
        """
        e = np.atleast_1d(np.asarray(energies_kev, dtype=float))
        y = (e / self.critical_energy_kev)[:, None]
        x2 = (self.gamma * np.atleast_1d(np.asarray(psi_urad, dtype=float)) * 1e-6)[None, :] ** 2
        xi = 0.5 * y * (1.0 + x2) ** 1.5
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            core = kv(2.0 / 3.0, xi) ** 2 + x2 / (1.0 + x2) * kv(1.0 / 3.0, xi) ** 2
        core = np.nan_to_num(core, nan=0.0, posinf=0.0)
        return 1.327e13 * self.ring_energy_gev ** 2 * self.current_a * y ** 2 * (1.0 + x2) ** 2 * core

    def accepted_flux(self, energies_kev: np.ndarray, vertical_acceptance_urad: Optional[float] = None,
                      vertical_offset_urad: float = 0.0, n_angles: int = 41) -> np.ndarray:
        """Flux per mrad (horizontal) per 0.1 % BW inside a vertical window.

        The window is ``vertical_offset_urad +/- vertical_acceptance_urad / 2``
        in the source vertical angle.  ``None`` means the full opening angle,
        i.e. :meth:`spectral_flux`.  For an acceptance much smaller than
        1/gamma the result is the on-axis spectrum times the acceptance.
        """
        if vertical_acceptance_urad is None or not np.isfinite(vertical_acceptance_urad):
            return self.spectral_flux(energies_kev)
        half = 0.5 * float(vertical_acceptance_urad)
        psi = np.linspace(vertical_offset_urad - half, vertical_offset_urad + half, n_angles)
        return trapezoid(self.angular_flux(energies_kev, psi), psi * 1e-3, axis=1)


# =============================================================================
# Optics
# =============================================================================

@dataclass
class Mirror:
    """Grazing-incidence mirror, s-polarised Fresnel reflectivity.

    Parameters
    ----------
    material : str
        Compound formula of the reflecting surface, e.g. ``"Ir"``, ``"Si"``.
    density_g_cm3 : float
        Density of that material.
    grazing_angle_mrad : float
        Angle between the incident beam and the mirror surface.
    roughness_nm : float
        RMS surface roughness; applied as a Nevot-Croce factor.  Zero disables it.
    label : str
        Free-text name for plots.
    distance_m : float, optional
        Distance from the source.  Needed, with ``length_m``, for the mirror
        footprint to act as an aperture in :meth:`Beamline.acceptance`.
    length_m : float, optional
        Optical length.  The mirror accepts a beam of size
        ``length * sin(theta)`` in its deflection plane.
    plane : {"vertical", "horizontal"}
        Deflection plane.
    bending_radius_m : float
        Meridional radius of curvature (``inf`` for flat).  A concave mirror
        of radius ``R`` acts as a lens of focal length ``R sin(theta) / 2`` on
        the beam divergence, which changes how downstream apertures map back
        to the source.
    image_distance_m : float, optional
        For a focusing mirror (KB): distance to the focus.  The grazing angle
        then varies across the illuminated footprint by ``+/- s / (4 q)`` for a
        beam of size ``s``; :class:`Beamline` sets ``angle_spread_mrad`` from it.
    angle_spread_mrad : float
        Half-range of grazing angles over the illuminated footprint; the
        reflectivity is averaged uniformly over ``theta +/- spread``.
    """

    material: str
    density_g_cm3: float
    grazing_angle_mrad: float
    roughness_nm: float = 0.0
    label: str = ""
    distance_m: Optional[float] = None
    length_m: Optional[float] = None
    plane: str = "vertical"
    bending_radius_m: float = math.inf
    image_distance_m: Optional[float] = None
    angle_spread_mrad: float = 0.0

    def __post_init__(self):
        self.label = self.label or f"{self.material} mirror @ {self.grazing_angle_mrad:.3f} mrad"
        if self.plane not in ("vertical", "horizontal"):
            raise ValueError(f"plane must be 'vertical' or 'horizontal', got {self.plane!r}")

    @property
    def focal_length_m(self) -> float:
        """``R sin(theta) / 2`` (``inf`` for a flat mirror)."""
        if not np.isfinite(self.bending_radius_m):
            return math.inf
        return self.bending_radius_m * math.sin(self.grazing_angle_mrad * 1e-3) / 2.0

    @property
    def aperture_m(self) -> Optional[float]:
        """Beam size the footprint accepts, ``length sin(theta)``."""
        if self.length_m is None:
            return None
        return self.length_m * math.sin(self.grazing_angle_mrad * 1e-3)

    def transfer(self, energies_kev: np.ndarray) -> np.ndarray:
        """Reflectivity in [0, 1], averaged over ``angle_spread_mrad`` if set."""
        if self.angle_spread_mrad > 0:
            angles = np.linspace(self.grazing_angle_mrad - self.angle_spread_mrad,
                                 self.grazing_angle_mrad + self.angle_spread_mrad, 21)
            return np.mean([self._reflectivity(energies_kev, a) for a in angles], axis=0)
        return self._reflectivity(energies_kev, self.grazing_angle_mrad)

    def _reflectivity(self, energies_kev: np.ndarray, angle_mrad: float) -> np.ndarray:
        e = np.atleast_1d(np.asarray(energies_kev, dtype=float))
        theta = angle_mrad * 1e-3
        sin_t, cos_t = np.sin(theta), np.cos(theta)

        n = np.ones_like(e, dtype=complex)
        for i, ek in enumerate(e):
            try:
                n[i] = (xraylib.Refractive_Index_Re(self.material, ek, self.density_g_cm3)
                        + 1j * xraylib.Refractive_Index_Im(self.material, ek, self.density_g_cm3))
            except ValueError:
                n[i] = 1.0

        # q = sqrt(n^2 - cos^2(theta));  r_s = (sin - q) / (sin + q)
        q = np.sqrt(n ** 2 - cos_t ** 2)
        r = np.abs((sin_t - q) / (sin_t + q)) ** 2

        if self.roughness_nm > 0:
            # Nevot-Croce acts on the reflection *amplitude*:
            #   r_s -> r_s * exp(-2 kz0 kz1 sigma^2),
            # so the intensity factor is exp(-4 kz0 Re(kz1) sigma^2), with
            # kz0 = k0 sin(theta) and kz1 = k0 q.  Below the critical angle
            # Re(q) << sin(theta) (evanescent wave) and the factor -> 1; the
            # Debye-Waller limit exp(-4 kz0^2 sigma^2) holds only well above it.
            lam_nm = 1.23984193 / e                      # keV -> nm
            k0 = 2.0 * np.pi / lam_nm
            r = r * np.exp(-4.0 * k0 ** 2 * sin_t * np.real(q) * self.roughness_nm ** 2)

        return np.clip(r.real, 0.0, 1.0)


@dataclass
class Filter:
    """Transmission through a slab of material (window, filter, air path).

    Parameters
    ----------
    material : str
        Compound formula, e.g. ``"Be"``, ``"Al"``, ``"C"``, ``"Kapton"`` as
        ``"C22H10N2O5"``.
    density_g_cm3 : float
    thickness_um : float
    label : str
    """

    material: str
    density_g_cm3: float
    thickness_um: float
    label: str = ""

    def __post_init__(self):
        self.label = self.label or f"{self.thickness_um:g} um {self.material}"

    def transfer(self, energies_kev: np.ndarray) -> np.ndarray:
        """Transmission in [0, 1], from the Beer-Lambert law."""
        e = np.atleast_1d(np.asarray(energies_kev, dtype=float))
        t_cm = self.thickness_um * 1e-4
        mu = np.zeros_like(e)
        for i, ek in enumerate(e):
            try:
                mu[i] = xraylib.CS_Total_CP(self.material, ek) * self.density_g_cm3
            except ValueError:
                mu[i] = np.inf
        return np.exp(-mu * t_cm)


@dataclass
class Slit:
    """A pair of slits: an aperture without spectral effect.

    Parameters
    ----------
    distance_m : float
        Distance from the source.
    vertical_gap_mm, horizontal_gap_mm : float, optional
        Gaps; ``None`` means open (no limit) in that plane.
    label : str
    """

    distance_m: float
    vertical_gap_mm: Optional[float] = None
    horizontal_gap_mm: Optional[float] = None
    label: str = ""

    def __post_init__(self):
        v = "open" if self.vertical_gap_mm is None else f"{self.vertical_gap_mm:g}"
        h = "open" if self.horizontal_gap_mm is None else f"{self.horizontal_gap_mm:g}"
        self.label = self.label or f"slit @ {self.distance_m:g} m ({v} x {h} mm)"

    def transfer(self, energies_kev: np.ndarray) -> np.ndarray:
        return np.ones_like(np.atleast_1d(np.asarray(energies_kev, dtype=float)))


OpticalElement = Union[Mirror, Filter, Slit]


def per_unit_energy(energies_kev: np.ndarray, flux_01bw: np.ndarray) -> np.ndarray:
    """Convert a per-0.1 %-bandwidth flux into a per-keV spectral density.

    ``dN/dE = N_0.1%BW / (1e-3 * E)``: a fixed *relative* bandwidth spans an
    energy interval proportional to E.  Integrals over ``dE`` need this form.
    """
    e = np.asarray(energies_kev, dtype=float)
    return np.asarray(flux_01bw, dtype=float) / (1e-3 * np.maximum(e, 1e-12))


# =============================================================================
# Air
# =============================================================================

#: Dry air as an atom-count formula, from the standard volume composition
#: (N2 78.084 %, O2 20.946 %, Ar 0.934 %, CO2 0.0417 %) converted to atoms per
#: 100 molecules.  Reproduces the accepted mass fractions
#: N 75.5 %, O 23.2 %, Ar 1.29 %, C 0.017 %.
AIR_FORMULA = "N156.168O41.975Ar0.934C0.0417"

#: Molar mass of dry air, g/mol.
AIR_MOLAR_MASS = 28.9647

#: Molar gas constant, J/(mol K).
GAS_CONSTANT = 8.31446261815324


def air_density(temperature_c: float = 20.0, pressure_hpa: float = 1013.25) -> float:
    """Dry-air density in g/cm^3 from the ideal gas law.

    ``rho = P M / (R T)``.  Humidity is neglected: moist air is slightly *less*
    dense than dry air (water is lighter than N2/O2), so at 50 % relative
    humidity and 20 C this overestimates the density, and hence the absorption,
    by a few parts per thousand.

    Note that pressure is the *local* value, not sea-level corrected: at the
    ESRF (~210 m) the ambient pressure is nearer 990 hPa than 1013 hPa, which is
    a 2.5 % change in density and therefore in optical depth.
    """
    return (pressure_hpa * 100.0) * (AIR_MOLAR_MASS * 1e-3) / (
        GAS_CONSTANT * (temperature_c + 273.15)) * 1e-3


def air_path(length_m: float, temperature_c: float = 20.0,
             pressure_hpa: float = 1013.25) -> Filter:
    """A :class:`Filter` representing a path through ambient air.

    A long air path is a strong low-energy filter and cannot be neglected on a
    white-beam line: 3 m of air at ambient conditions transmits ~1e-7 at 5 keV,
    1.7 % at 8 keV and 12 % at 10 keV, so it sets the practical lower bound of
    the usable band.

    Parameters
    ----------
    length_m : float
        Path length in metres.
    temperature_c, pressure_hpa : float
        Ambient conditions, used to compute the density.
    """
    rho = air_density(temperature_c, pressure_hpa)
    return Filter(material=AIR_FORMULA, density_g_cm3=rho,
                  thickness_um=length_m * 1e6,
                  label=f"{length_m:g} m air ({temperature_c:g} C, {pressure_hpa:g} hPa)")


# =============================================================================
# Assembled beamline
# =============================================================================

@dataclass
class Beamline:
    """A source plus an ordered list of optical elements.

    Elements with a position (:class:`Slit`, :class:`Mirror` with
    ``distance_m``) also act as apertures.  :meth:`acceptance` maps each of them
    back to an angle at the source, through the focusing action of the bent
    mirrors upstream (paraxial optics), and the smallest one in the vertical
    plane sets the window of source angles whose spectrum is used.

    Parameters
    ----------
    source : BendingMagnetSource
    optics : list
        Mirrors, filters and slits.  The order matters only for labelling;
        positions are taken from ``distance_m``.
    sample_distance_m : float, optional
        Source-to-sample distance (for :meth:`trace` and :meth:`describe`).
    vertical_acceptance : "auto", "full" or float
        ``"auto"``: from the apertures (full opening angle if there are none);
        ``"full"``: vertically integrated spectrum (``G1``), the pre-2026
        behaviour; a number: full vertical acceptance in urad.
    vertical_offset_urad : float
        Centre of the accepted window relative to the orbit plane (0 when the
        apertures are aligned on the beam).
    settings : dict
        Free-form provenance (e.g. the motor positions read by
        :func:`bm32_from_h5`).
    """

    source: BendingMagnetSource
    optics: List[OpticalElement] = field(default_factory=list)
    sample_distance_m: Optional[float] = None
    vertical_acceptance: Union[str, float] = "auto"
    vertical_offset_urad: float = 0.0
    settings: Dict[str, object] = field(default_factory=dict)

    # -- geometry ---------------------------------------------------------------
    def _positioned(self) -> List[OpticalElement]:
        return sorted((o for o in self.optics if getattr(o, "distance_m", None) is not None),
                      key=lambda o: o.distance_m)

    def trace(self, plane: str = "vertical") -> List[Tuple[Optional[OpticalElement], float]]:
        """Paraxial height per unit source angle (m/rad) at each positioned element.

        A ray leaving the source on axis with unit angle is propagated; every
        mirror deflecting in ``plane`` with a finite bending radius reduces the
        angle by ``h / f`` (concave mirror, ``f = R sin(theta) / 2``).  The last
        entry, labelled ``None``, is the sample if ``sample_distance_m`` is set.
        """
        h, a, z = 0.0, 1.0, 0.0
        out: List[Tuple[Optional[OpticalElement], float]] = []
        for o in self._positioned():
            h += a * (o.distance_m - z)
            z = o.distance_m
            out.append((o, h))
            if isinstance(o, Mirror) and o.plane == plane and np.isfinite(o.focal_length_m):
                a -= h / o.focal_length_m
        if self.sample_distance_m is not None:
            out.append((None, h + a * (self.sample_distance_m - z)))
        return out

    def acceptance(self) -> Dict[str, dict]:
        """Full angular acceptance at the source (urad) of every aperture, per plane.

        Returns ``{"vertical": {...}, "horizontal": {...}}`` with ``limits``
        ({label: urad}), ``acceptance_urad`` (the smallest) and ``limited_by``.
        Slits act through their gaps, mirrors through ``length sin(theta)`` in
        their deflection plane.
        """
        res = {}
        for plane in ("vertical", "horizontal"):
            lim: Dict[str, float] = {}
            for o, m in self.trace(plane):
                if o is None or m <= 0:
                    continue
                size = None
                if isinstance(o, Slit):
                    gap = o.vertical_gap_mm if plane == "vertical" else o.horizontal_gap_mm
                    size = None if gap is None else gap * 1e-3
                elif isinstance(o, Mirror) and o.plane == plane:
                    size = o.aperture_m
                if size is not None:
                    key = o.label if o.label not in lim else f"{o.label} @ {o.distance_m:g} m"
                    lim[key] = size / m * 1e6
            if lim:
                k = min(lim, key=lim.get)
                res[plane] = dict(limits=lim, acceptance_urad=lim[k], limited_by=k)
            else:
                res[plane] = dict(limits={}, acceptance_urad=math.inf, limited_by=None)
        return res

    def effective_vertical_acceptance(self) -> Optional[float]:
        """Full vertical acceptance used for the spectrum (urad), ``None`` = full."""
        if isinstance(self.vertical_acceptance, str):
            if self.vertical_acceptance == "full":
                return None
            if self.vertical_acceptance != "auto":
                raise ValueError("vertical_acceptance must be 'auto', 'full' or a number")
            a = self.acceptance()["vertical"]["acceptance_urad"]
            return None if not np.isfinite(a) else float(a)
        return float(self.vertical_acceptance)

    def resolved_optics(self) -> List[OpticalElement]:
        """The optics with the KB angle spreads filled in.

        For a positioned mirror with ``image_distance_m`` and no explicit
        ``angle_spread_mrad``, the
        illuminated beam size ``s`` is the accepted angle in the mirror's plane
        times the height-per-angle at the mirror, or the mirror aperture if
        nothing upstream limits it, and the spread is ``s / (4 q)``.
        """
        acc = self.acceptance()
        traces = {p: {id(o): m for o, m in self.trace(p) if o is not None} for p in ("vertical", "horizontal")}
        out = []
        for o in self.optics:
            if isinstance(o, Mirror) and o.angle_spread_mrad == 0.0 and o.distance_m is not None:
                q = o.image_distance_m            # only mirrors declared as focusing get a spread
                if q is not None and q > 0:
                    a = acc[o.plane]["acceptance_urad"]
                    size = a * 1e-6 * traces[o.plane][id(o)] if np.isfinite(a) else (o.aperture_m or 0.0)
                    if o.aperture_m is not None:
                        size = min(size, o.aperture_m)
                    o = replace(o, angle_spread_mrad=1e3 * size / (4.0 * q))
            out.append(o)
        return out

    def describe(self) -> Dict[str, object]:
        """Summary: acceptances, limiting apertures, KB spreads, apparent source."""
        acc = self.acceptance()
        tr = self.trace("vertical")
        spreads = {o.label: o.angle_spread_mrad for o in self.resolved_optics()
                   if isinstance(o, Mirror) and o.angle_spread_mrad > 0}
        return dict(vertical_acceptance_urad=acc["vertical"]["acceptance_urad"],
                    vertical_limited_by=acc["vertical"]["limited_by"],
                    horizontal_acceptance_urad=acc["horizontal"]["acceptance_urad"],
                    horizontal_limited_by=acc["horizontal"]["limited_by"],
                    vertical_limits_urad=acc["vertical"]["limits"],
                    horizontal_limits_urad=acc["horizontal"]["limits"],
                    kb_angle_spread_mrad=spreads,
                    vertical_height_per_urad_um={(o.label if o is not None else "sample"): m for o, m in tr},
                    used_vertical_acceptance_urad=self.effective_vertical_acceptance(),
                    vertical_offset_urad=self.vertical_offset_urad)

    # -- spectrum ---------------------------------------------------------------
    def spectrum(self, energies_kev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, np.ndarray]]]:
        """Propagate the accepted source spectrum through the optics.

        Returns
        -------
        raw : (n,) array
            Source flux inside the accepted vertical window, per mrad
            (horizontal) per 0.1 % BW, before any optic.
        transmitted : (n,) array
            Flux at the sample.
        transfers : list of (label, array)
            Each mirror's or filter's transfer function, in order (slits
            have none).
        """
        e = np.asarray(energies_kev, dtype=float)
        raw = self.source.accepted_flux(e, self.effective_vertical_acceptance(), self.vertical_offset_urad)
        transmitted = raw.copy()
        transfers = []
        for optic in self.resolved_optics():
            if isinstance(optic, Slit):
                continue
            t = optic.transfer(e)
            transfers.append((optic.label, t))
            transmitted = transmitted * t
        return raw, transmitted, transfers

    def spectral_density(self, energies_kev: np.ndarray) -> np.ndarray:
        """Flux at the sample per unit energy, photons/s/mrad/keV.

        The quantity to integrate over ``dE``; see :func:`per_unit_energy`.
        """
        e = np.asarray(energies_kev, dtype=float)
        _, transmitted, _ = self.spectrum(e)
        return per_unit_energy(e, transmitted)

    def band_limits(self, energies_kev: np.ndarray, frac: float = 0.01) -> Tuple[float, float]:
        """Energies where the transmitted flux first and last exceed ``frac`` of its peak.

        A compact way to state the usable white-beam band.
        """
        e = np.asarray(energies_kev, dtype=float)
        _, phi, _ = self.spectrum(e)
        above = np.flatnonzero(phi >= frac * phi.max())
        if above.size == 0:
            raise ValueError("Transmitted flux never reaches the requested fraction of its peak.")
        return float(e[above[0]]), float(e[above[-1]])


# =============================================================================
# BM32
# =============================================================================

#: BM32 layout: distances from the source (m) and optical lengths (m), as
#: given by the beamline (Oct 2026).  s1 primary slits; m1, m2 bent Ir mirrors
#: deflecting vertically; s2 guard slits against scattered radiation; s3 slits
#: defining the KB illumination; KB-V (vertical) and KB-H (horizontal)
#: elliptical Ir mirrors; sample.
BM32_LAYOUT: Dict[str, float] = dict(
    s1=26.368, m1=28.309, m2=31.732, s2=35.370, s3=44.485, kbv=44.645, kbh=44.880, sample=45.0,
    m1_length=1.2, m2_length=1.2, kbv_length=0.30, kbh_length=0.15,
)


def BM32_WHITE_BEAM(air_path_m: float = 3.3, temperature_c: float = 20.0,
                    pressure_hpa: float = 1013.25, current_a: float = 0.2,
                    magnetic_field_t: float = 0.86, *,
                    vg1: Optional[float] = 0.3, hg1: Optional[float] = 0.2,
                    vg2: Optional[float] = None, hg2: Optional[float] = None,
                    vg3: Optional[float] = 0.2, hg3: Optional[float] = 0.2,
                    m1_angle_mrad: float = 3.062, m2_angle_mrad: float = 2.551,
                    m1_radius_km: float = 71.4, m2_radius_km: float = 145.7,
                    kbv_angle_mrad: float = 3.0, kbh_angle_mrad: float = 3.0,
                    kb_angle_spread: bool = True,
                    vertical_acceptance: Union[str, float] = "auto",
                    vertical_offset_urad: float = 0.0) -> Beamline:
    """BM32 white-beam model (see module caveats).

    The defaults are representative of the 2026 Laue maps; use
    :func:`bm32_from_h5` to take the slit gaps, mirror angles, bending radii
    and ring current of a given scan.  The spectrum is insensitive to the exact
    gaps as long as the vertical acceptance stays well below ~30 urad.

    Parameters
    ----------
    air_path_m : float
        Length of the **incident** air path, in metres (3.3 m measured on
        BM32).  Set to 0 to remove it.
        This is the dominant low-energy filter on the line and it is what sets
        the ~5 keV lower bound of the usable band; the 1 mm of Be alone would
        pass energies well below that.

        Do not confuse it with the **detection** air path (sample to detector
        window), which attenuates the *emitted* lines instead and belongs to a
        separate detector-response model.
    temperature_c, pressure_hpa : float
        Ambient conditions for the air density.
    current_a : float
        Ring current in amperes (scales the absolute flux only).
    magnetic_field_t : float
        Source field.  Sets the critical energy.
    vg1, hg1, vg2, hg2, vg3, hg3 : float or None
        Vertical / horizontal gaps of slits s1, s2, s3 in mm (motors of the
        same names); ``None`` = open.
    m1_angle_mrad, m2_angle_mrad : float
        Grazing angles of m1, m2 (motors ``ma1``, ``ma2``).
    m1_radius_km, m2_radius_km : float
        Bending radii of m1, m2 (motors ``mc1``, ``mc2``); ``inf`` for flat.
    kbv_angle_mrad, kbh_angle_mrad : float
        KB grazing angles (not recorded in the scan files; 3 mrad assumed).
    kb_angle_spread : bool
        Average the KB reflectivities over the grazing-angle spread of their
        illuminated footprints.
    vertical_acceptance, vertical_offset_urad
        See :class:`Beamline`.  ``"full"`` restores the vertically integrated
        (``G1``) spectrum.
    """
    L = BM32_LAYOUT
    q_v = L["sample"] - L["kbv"] if kb_angle_spread else None
    q_h = L["sample"] - L["kbh"] if kb_angle_spread else None
    optics: List[OpticalElement] = [
        Slit(L["s1"], vg1, hg1, label="s1"),
        Mirror("Ir", 22.56, m1_angle_mrad, label=f"m1 Ir @ {m1_angle_mrad:.3f} mrad", distance_m=L["m1"],
               length_m=L["m1_length"], bending_radius_m=m1_radius_km * 1e3),
        Mirror("Ir", 22.56, m2_angle_mrad, label=f"m2 Ir @ {m2_angle_mrad:.3f} mrad", distance_m=L["m2"],
               length_m=L["m2_length"], bending_radius_m=m2_radius_km * 1e3),
        Filter("Be", 1.85, 500.0),
        Filter("Be", 1.85, 500.0),
        Slit(L["s2"], vg2, hg2, label="s2"),
        Slit(L["s3"], vg3, hg3, label="s3"),
        Mirror("Ir", 22.56, kbv_angle_mrad, label=f"KB-V Ir @ {kbv_angle_mrad:.3f} mrad", distance_m=L["kbv"],
               length_m=L["kbv_length"], plane="vertical", image_distance_m=q_v),
        Mirror("Ir", 22.56, kbh_angle_mrad, label=f"KB-H Ir @ {kbh_angle_mrad:.3f} mrad", distance_m=L["kbh"],
               length_m=L["kbh_length"], plane="horizontal", image_distance_m=q_h),
    ]
    if air_path_m > 0:
        optics.append(air_path(air_path_m, temperature_c, pressure_hpa))
    return Beamline(
        source=BendingMagnetSource(ring_energy_gev=6.04, magnetic_field_t=magnetic_field_t, current_a=current_a),
        optics=optics, sample_distance_m=L["sample"],
        vertical_acceptance=vertical_acceptance, vertical_offset_urad=vertical_offset_urad,
        settings=dict(vg1=vg1, hg1=hg1, vg2=vg2, hg2=hg2, vg3=vg3, hg3=hg3, ma1=m1_angle_mrad,
                      ma2=m2_angle_mrad, mc1_km=m1_radius_km, mc2_km=m2_radius_km,
                      kbv_angle_mrad=kbv_angle_mrad, kbh_angle_mrad=kbh_angle_mrad,
                      current_a=current_a, air_path_m=air_path_m),
    )


#: BLISS positioner names read by :func:`bm32_from_h5`, with the keyword they set.
_BM32_MOTORS = dict(vg1="vg1", hg1="hg1", vg2="vg2", hg2="hg2", vg3="vg3", hg3="hg3",
                    ma1="m1_angle_mrad", ma2="m2_angle_mrad", mc1="m1_radius_km", mc2="m2_radius_km")


def bm32_from_h5(h5_path: Union[str, Path], scan: Optional[str] = None, **overrides) -> Beamline:
    """BM32 beamline with the settings recorded in a BLISS scan file.

    Reads the slit gaps (``vg1 hg1 vg2 hg2 vg3 hg3``, mm), the m1/m2 grazing
    angles (``ma1 ma2``, mrad) and bending radii (``mc1 mc2``, km) from
    ``<scan>/instrument/positioners`` and the ring current from
    ``<scan>/instrument/machine/current`` (mA).  Values that are missing or not
    numeric (BLISS writes ``ERR`` for disconnected motors) keep their defaults,
    with a warning.  Keyword arguments override anything, e.g. ``air_path_m``,
    ``kbv_angle_mrad`` or ``vertical_acceptance``.

    Parameters
    ----------
    h5_path : path
    scan : str, optional
        Scan entry, e.g. ``"1.1"``; the first entry by default.
    """
    import h5py

    kw: Dict[str, object] = {}
    with h5py.File(h5_path, "r") as f:
        entry = scan or sorted(f.keys())[0]
        ins = f[entry]["instrument"]
        pos = ins["positioners"] if "positioners" in ins else {}
        for motor, key in _BM32_MOTORS.items():
            if motor not in pos:
                logger.warning("bm32_from_h5: positioner %r not found; default kept.", motor)
                continue
            try:
                kw[key] = float(np.atleast_1d(pos[motor][()])[0])
            except (TypeError, ValueError):
                logger.warning("bm32_from_h5: positioner %r is not numeric; default kept.", motor)
        try:
            kw["current_a"] = float(ins["machine"]["current"][()]) * 1e-3
        except (KeyError, TypeError, ValueError):
            logger.warning("bm32_from_h5: ring current not found; default kept.")
    kw.update(overrides)
    bl = BM32_WHITE_BEAM(**kw)
    bl.settings["source_file"] = str(h5_path)
    bl.settings["scan"] = entry
    return bl
