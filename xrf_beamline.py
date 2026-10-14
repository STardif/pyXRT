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
* ``spectral_flux`` returns photons/s/mrad(horizontal)/0.1 % bandwidth, using
  the standard prefactor ``1.327e13 * E[GeV]^2 * I[A] * G1(y)``.  **The current
  is in amperes.**  (The variant written as ``2.457e13 * E[GeV] * I[A] * G1(y)``
  is the same expression with one factor of E folded differently; feeding it a
  value in mA overestimates the flux by 1000.  The absolute scale cancels in
  every ratio computed here, but it matters if you ever quote absolute flux.)
* ``G1`` is the **vertically integrated** spectral shape, so it is the right
  weighting only if the beamline accepts the full natural vertical opening
  angle.  With vertical collimation tighter than ~1/gamma you need the
  angle-resolved form instead.
* Mirror reflectivity is s-polarised Fresnel with an optional Nevot-Croce
  roughness factor.  Synchrotron radiation is predominantly horizontally
  polarised and the mirrors here deflect vertically, so s-polarisation is the
  relevant case; at grazing incidence the two are indistinguishable anyway.
* The default optics list below is taken from the earlier BM32 analysis
  notebook and is **unverified** — mirror coating, grazing angles, window
  thicknesses and the source field should be confirmed against the beamline
  documentation before any quantitative use.

Authors: Samuel Tardif, with Claude (Anthropic).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Union

import numpy as np
import xraylib
from scipy.integrate import quad
from scipy.special import kv

logger = logging.getLogger(__name__)
xraylib.SetErrorMessages(0)

__all__ = [
    "BendingMagnetSource",
    "Mirror",
    "Filter",
    "Beamline",
    "BM32_WHITE_BEAM",
    "air_path",
    "air_density",
    "AIR_FORMULA",
]


# =============================================================================
# Source
# =============================================================================

@dataclass
class BendingMagnetSource:
    """Schwinger spectral flux of a bending-magnet (or short wiggler) source.

    Parameters
    ----------
    ring_energy_gev : float
        Storage-ring energy.  6.04 GeV for ESRF-EBS.
    magnetic_field_t : float
        Dipole field.  Sets the critical energy, to which the high-energy
        spectrum is exponentially sensitive — confirm it for your source
        (an EBS two-pole wiggler is not the same as a standard EBS dipole).
    current_a : float
        Ring current in **amperes** (0.2 A = 200 mA).
    """

    ring_energy_gev: float = 6.04
    magnetic_field_t: float = 0.85
    current_a: float = 0.2

    @property
    def critical_energy_kev(self) -> float:
        """``Ec = 0.665 * E[GeV]^2 * B[T]`` in keV."""
        return 0.665 * self.ring_energy_gev ** 2 * self.magnetic_field_t

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
        """Flux in photons/s/mrad(horizontal)/0.1 % bandwidth."""
        e = np.atleast_1d(np.asarray(energies_kev, dtype=float))
        y = e / self.critical_energy_kev
        shape = np.array([self._g1(float(yy)) for yy in y])
        prefactor = 1.327e13 * self.ring_energy_gev ** 2 * self.current_a
        return prefactor * shape


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
    """

    material: str
    density_g_cm3: float
    grazing_angle_mrad: float
    roughness_nm: float = 0.0
    label: str = ""

    def __post_init__(self):
        self.label = self.label or f"{self.material} mirror @ {self.grazing_angle_mrad:.3f} mrad"

    def transfer(self, energies_kev: np.ndarray) -> np.ndarray:
        """Reflectivity in [0, 1]."""
        e = np.atleast_1d(np.asarray(energies_kev, dtype=float))
        theta = self.grazing_angle_mrad * 1e-3
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
            # Nevot-Croce: R -> R * exp(-2 kz kz' sigma^2), approximated with
            # kz = kz' = (2 pi / lambda) sin(theta).
            lam_nm = 1.23984193 / e                      # keV -> nm
            kz = 2.0 * np.pi * sin_t / lam_nm
            r = r * np.exp(-2.0 * (kz * self.roughness_nm) ** 2)

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


OpticalElement = Union[Mirror, Filter]


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
    """A source plus an ordered list of optical elements."""

    source: BendingMagnetSource
    optics: List[OpticalElement] = field(default_factory=list)

    def spectrum(self, energies_kev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, np.ndarray]]]:
        """Propagate the source spectrum through the optics.

        Returns
        -------
        raw : (n,) array
            Source flux before any optic.
        transmitted : (n,) array
            Flux at the sample.
        transfers : list of (label, array)
            Each element's transfer function, in order.
        """
        e = np.asarray(energies_kev, dtype=float)
        raw = self.source.spectral_flux(e)
        transmitted = raw.copy()
        transfers = []
        for optic in self.optics:
            t = optic.transfer(e)
            transfers.append((optic.label, t))
            transmitted = transmitted * t
        return raw, transmitted, transfers

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


#: BM32 white-beam configuration transcribed from the earlier analysis notebook.
#:
#: UNVERIFIED — four Ir-coated mirrors and two 500 um Be windows, with the
#: grazing angles as written there.  Confirm against beamline documentation
#: before using any absolute number derived from it.  The sample-side air path
#: and the detector window are *not* included: they belong to the detection
#: path, not the excitation path.
def BM32_WHITE_BEAM(air_path_m: float = 3.0, temperature_c: float = 20.0,
                    pressure_hpa: float = 1013.25, current_a: float = 0.2,
                    magnetic_field_t: float = 0.85) -> Beamline:
    """Default BM32 white-beam model (see module caveats).

    Parameters
    ----------
    air_path_m : float
        Length of the **incident** air path, in metres.  Set to 0 to remove it.
        This is the dominant low-energy filter on the line and it is what sets
        the ~5 keV lower bound of the usable band; the 1 mm of Be alone would
        pass energies well below that.

        Do not confuse it with the **detection** air path (sample to detector
        window), which attenuates the *emitted* lines instead and belongs to a
        separate detector-response model.
    temperature_c, pressure_hpa : float
        Ambient conditions for the air density.
    current_a : float
        Ring current in amperes.
    magnetic_field_t : float
        Source field.  Sets the critical energy.
    """
    optics: List[OpticalElement] = [
        Mirror("Ir", 22.56, 3.062),
        Mirror("Ir", 22.56, 2.551),
        Filter("Be", 1.85, 500.0),
        Filter("Be", 1.85, 500.0),
        Mirror("Ir", 22.56, 3.000),
        Mirror("Ir", 22.56, 3.000),
    ]
    if air_path_m > 0:
        optics.append(air_path(air_path_m, temperature_c, pressure_hpa))
    return Beamline(
        source=BendingMagnetSource(ring_energy_gev=6.04,
                                   magnetic_field_t=magnetic_field_t,
                                   current_a=current_a),
        optics=optics,
    )
