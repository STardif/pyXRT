"""
Regenerate every figure and every number of the article
``docs/xrf_quantification/xrf_quantification.tex`` from the raw data.

Runs the same processing chain as ``fluo_analysis_clean.ipynb`` on the
InPoSi_v2_center map (50 nm InP / 1.9 um (In,Ga)As / Si), writes the figures to
``figures/`` (PDF for the article, PNG for previews) and all quoted numbers to
``results.json``.

Usage (from the repository root or anywhere)::

    python docs/xrf_quantification/make_figures.py

Needs the data file ``InPoSi_v2_center.h5`` at the repository root.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import h5py
import xraylib
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.integrate import trapezoid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import xrf_beamline as XB                                   # noqa: E402
import xrf_fit as XF                                        # noqa: E402
import xrf_matrix as XM                                     # noqa: E402
from bliss_hdf5_reader import BlissMeshScan                 # noqa: E402

logging.basicConfig(level=logging.WARNING)
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
R: dict = {}                       # every number quoted in the article
T0 = time.time()

# ----------------------------------------------------------------------------- style
TEXTW = 6.3                        # inches, single-column A4 text width
mpl.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out", "axes.titlelocation": "left",
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    "lines.linewidth": 1.0,
})
C = {"data": "0.15", "model": "#c0392b", "bkg": "0.55", "In": "#1f77b4", "Ga": "#e67e22",
     "As": "#2ca02c", "Cu": "#8c564b", "Ni": "#9467bd", "Ca": "#7f7f7f", "Ar": "#e377c2",
     "esc": "#17becf", "pile": "#bcbd22", "accent": "#2471a3"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def panel(ax, letter):
    ax.text(-0.10, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=9,
            va="bottom", ha="right")


def greek(label):
    return label.replace("a1", "α₁").replace("b1", "β₁")


# ----------------------------------------------------------------------------- configuration
H5 = ROOT / "InPoSi_v2_center.h5"
MCA, MON, LT, RT = "xia_det0", "mon", "xia_det0_live_time", "xia_det0_elapsed_time"
CAL = XF.Calibration(gain=10.032, offset=-31.263)
DET = XF.Detector.from_fwhm(135.0, 5900.0, "Si", 0.12, 5.0, sensor_thickness_um=450.0,
                            window_material="C", window_thickness_um=1.0, air_path_cm=20.0)
GEOM = XM.Geometry(40.0, 40.0)
ELEMENTS = ["In", "Ga", "As", "Cu", "Ni", "Ca", "Ar"]
FAMILIES = ("K", "L")
ROI_EV = (2900.0, 12600.0)
PILEUP_ROI_EV = (2900.0, 20300.0)
BKG = 3
MIN_REL = 0.005
E = np.arange(3.0, 40.0 + 1e-9, 0.010)
STACK = XM.Stack([XM.Layer("InP", 4.81, 0.05, "cap"),
                  XM.Layer("In0.5Ga0.5As", 5.5, 1.9, "film"),
                  XM.Layer("Si", 2.33, None, "substrate")])
TARGET, REF, ANALYTES = 1, "As K", ["Ga K", "In L"]

LINES = {"In La1": (49, xraylib.L3M5_LINE), "Ga Ka1": (31, xraylib.KL3_LINE),
         "As Ka1": (33, xraylib.KL3_LINE), "As Kb1": (33, xraylib.KM3_LINE)}
LINE_E = {k: xraylib.LineEnergy(z, m) for k, (z, m) in LINES.items()}

# ----------------------------------------------------------------------------- data
scan = BlissMeshScan(h5_filepath=H5, scan_number=1, verbose=False)
scan.read_data(extra_counters=[MCA, MON, LT, RT, f"{MCA}_fractional_dead_time"])
spectra = scan.counters[MCA].astype(float)
n_pix, n_ch = spectra.shape
n_slow, n_fast = scan.scan_shape
mon, lt, rt = scan.counters[MON], scan.counters[LT], scan.counters[RT]
fdt = scan.counters[f"{MCA}_fractional_dead_time"]
with h5py.File(H5, "r") as f:
    m = f["1.1/measurement"]
    icr = m[f"{MCA}_trigger_count_rate"][()]
    ocr = m[f"{MCA}_event_count_rate"][()]
    epoch = m["epoch"][()]
corrected, scale = XF.normalise_spectra(spectra, monitor=mon, live_time=lt)
avg, sem, n_used = XF.average_spectrum(corrected)
cnt_err = XF.counting_error(spectra, scale)
ROI = CAL.channel_range(*ROI_EV, n_ch)
ROI_PP = CAL.channel_range(*PILEUP_ROI_EV, n_ch)
Ech = CAL.energy(np.arange(n_ch))
step_um = abs(scan.fast_step_size) * 1000.0
R["data"] = dict(n_pixels=int(n_pix), shape=[int(n_slow), int(n_fast)], step_um=step_um,
                 count_time_s=float(scan.count_time),
                 counts_per_pixel=float(spectra.sum(1).mean()),
                 total_counts=float(spectra.sum()),
                 icr_kcps=[float(icr.min() / 1e3), float(icr.mean() / 1e3), float(icr.max() / 1e3)],
                 ocr_kcps_mean=float(ocr.mean() / 1e3),
                 dead_time=[float(fdt.min()), float(np.median(fdt)), float(fdt.max())],
                 live_time_s=[float(lt.min()), float(lt.max())],
                 real_time_s_mean=float(rt.mean()),
                 monitor_spread_pct=float(100 * np.ptp(mon) / mon.mean()),
                 scan_duration_s=float(epoch[-1] - epoch[0]),
                 roi_channels=list(map(int, ROI)), calibration=[CAL.gain, CAL.offset],
                 cnt_err_over_peak=float(np.median(cnt_err[ROI[0]:ROI[1]]) / avg[ROI[0]:ROI[1]].max()),
                 sem_over_peak=float(np.median(sem[ROI[0]:ROI[1]]) / avg[ROI[0]:ROI[1]].max()))
print(f"data loaded ({time.time()-T0:.0f}s)")

# ----------------------------------------------------------------------------- beam
bl = XB.bm32_from_h5(H5)                                     # slits, mirrors, current of this scan
bl_full = XB.bm32_from_h5(H5, vertical_acceptance="full", kb_angle_spread=False)   # pre-2026 model
raw_flux, phi, transfers = bl.spectrum(E)
raw_full, phi_full, _ = bl_full.spectrum(E)
dens = XB.per_unit_energy(E, phi)
dens_full = XB.per_unit_energy(E, phi_full)
TRANS = np.prod([t for _, t in transfers], axis=0)            # optics incl. KB angle spread
INFO = bl.describe()
with h5py.File(H5, "r") as f:
    FILL = f["1.1/instrument/machine/filling_mode"][()].decode()
src = bl.source
PSI = np.linspace(-600, 600, 2401)
ADEN = src.angular_flux(E, PSI)
fwhm_psi = {}
for e0 in (5, 10, 20, 30):
    w = ADEN[np.argmin(abs(E - e0))]
    fwhm_psi[str(e0)] = float(np.ptp(PSI[w >= w.max() / 2]))
Mv = INFO["vertical_height_per_urad_um"]                      # m per rad (= um per urad)
lab_kbv = next(k for k in Mv if k.startswith("KB-V")); lab_kbh = next(k for k in INFO["kb_angle_spread_mrad"] if k.startswith("KB-H"))
lab_m1 = next(k for k in Mv if k.startswith("m1")); lab_m2 = next(k for k in Mv if k.startswith("m2"))
mir = {o.label: o for o in bl.optics if isinstance(o, XB.Mirror)}
beam_v = INFO["vertical_acceptance_urad"] * 1e-6 * Mv[lab_kbv]
beam_h = INFO["horizontal_acceptance_urad"] * 1e-6 * XB.BM32_LAYOUT["kbh"]
R["beam"] = dict(field_t=src.magnetic_field_t, ec_kev=src.critical_energy_kev, gamma=src.gamma,
                 inv_gamma_urad=1e6 / src.gamma, fwhm_psi_urad=fwhm_psi,
                 current_ma=src.current_a * 1e3, filling_mode=FILL,
                 settings={k: v for k, v in bl.settings.items() if isinstance(v, (int, float))},
                 v_acc_urad=INFO["vertical_acceptance_urad"], v_limited_by=INFO["vertical_limited_by"],
                 h_acc_urad=INFO["horizontal_acceptance_urad"], h_limited_by=INFO["horizontal_limited_by"],
                 v_limits=INFO["vertical_limits_urad"], h_limits=INFO["horizontal_limits_urad"],
                 focal_m1_m=mir[lab_m1].focal_length_m, focal_m2_m=mir[lab_m2].focal_length_m,
                 height_per_urad_kbv=Mv[lab_kbv], divergence_gain=(Mv[lab_kbv] - Mv["s3"]) / (XB.BM32_LAYOUT["kbv"] - XB.BM32_LAYOUT["s3"]),
                 kb_spread_v_mrad=INFO["kb_angle_spread_mrad"][lab_kbv], kb_spread_h_mrad=INFO["kb_angle_spread_mrad"][lab_kbh],
                 kb_beam_v_mm=beam_v * 1e3, kb_beam_h_mm=beam_h * 1e3,
                 kb_footprint_v_mm=beam_v / 3e-3 * 1e3, kb_footprint_h_mm=beam_h / 3e-3 * 1e3,
                 band10_kev=list(bl.band_limits(E, 0.10)), band1_kev=list(bl.band_limits(E, 0.01)),
                 band10_full_kev=list(bl_full.band_limits(E, 0.10)),
                 peak_bw_kev=float(E[np.argmax(phi)]), peak_kev_kev=float(E[np.argmax(dens)]),
                 peak_kev_full=float(E[np.argmax(dens_full)]),
                 mean_kev=float(trapezoid(dens * E, E) / trapezoid(dens, E)),
                 mean_kev_full=float(trapezoid(dens_full * E, E) / trapezoid(dens_full, E)),
                 flux_20kev_ph_s_bw=float(phi[np.argmin(abs(E - 20))] * INFO["horizontal_acceptance_urad"] * 1e-3),
                 incident_air_m=3.3)


def k_gaas(weight):
    """Ga/As sensitivity factor (K lines, atomic) for a given spectral weighting."""
    s = {}
    for el, z in (("Ga", 31), ("As", 33)):
        tot = 0.0
        for mac in (xraylib.KL3_LINE, xraylib.KL2_LINE, xraylib.KM3_LINE, xraylib.KM2_LINE):
            tot += trapezoid(XM._cs_fluor(z, mac, E) * weight, E)
        s[el] = tot * xraylib.AtomicWeight(z)
    return s["As"] / s["Ga"]


R["beam"]["k_gaas_per_bw"] = float(k_gaas(phi))
R["beam"]["k_gaas_per_kev"] = float(k_gaas(dens))
R["beam"]["k_gaas_full"] = float(k_gaas(dens_full))
R["beam"]["k_gaas_no_kb_spread"] = float(k_gaas(XB.bm32_from_h5(H5, kb_angle_spread=False).spectral_density(E)))
ACC = np.array([1, 2, 5, 10, 20, 40, 80, 150, 300, 600, 1200])
OFFS = (0.0, 20.0, 40.0)
R["beam"]["acc_scan"] = dict(acc_urad=ACC.tolist(), offsets_urad=list(OFFS), k={
    str(o): [float(k_gaas(XB.per_unit_energy(E, src.accepted_flux(E, a, o, n_angles=241) * TRANS))) for a in ACC]
    for o in OFFS})
R["beam"]["off_scan"] = {str(o): float(k_gaas(XB.per_unit_energy(E, src.accepted_flux(E, INFO["vertical_acceptance_urad"], o) * TRANS)))
                         for o in (10.0, 25.0, 50.0)}

fig, axs = plt.subplots(1, 3, figsize=(TEXTW, 2.1), gridspec_kw={"wspace": 0.45})
ax = axs[0]
ax.plot(E, raw_full / raw_full.max(), ":", color="0.55", label="source, all ψ (G$_1$)")
ax.plot(E, raw_flux / raw_flux.max(), "--", color="0.2", label="source, accepted (ψ ≈ 0)")
ax.plot(E, phi / raw_flux.max(), color=C["model"], label="at the sample")
ax.set_xlabel("photon energy (keV)"); ax.set_ylabel("flux per 0.1 % BW (rel.)")
ax.set_ylim(0, 1.6)
ax.set_title("Bending-magnet spectrum"); ax.legend(loc="upper center", fontsize=5.6, ncol=1, borderaxespad=0.2); panel(ax, "a")
ax = axs[1]
comb, cnt = {}, {}
for lab, t in transfers:
    key = lab.replace(" mirror @ ", " @ ").replace(" um ", " µm ")
    comb[key] = comb.get(key, 1.0) * t; cnt[key] = cnt.get(key, 0) + 1
for key, t in comb.items():
    ax.semilogy(E, t, lw=0.9, label=(f"{cnt[key]} × " if cnt[key] > 1 else "") + key)
ax.semilogy(E, np.prod([t for _, t in transfers], axis=0), color="k", lw=1.3, label="total")
ax.set_ylim(1e-4, 2); ax.set_xlabel("photon energy (keV)"); ax.set_ylabel("transmission / reflectivity")
ax.set_title("Optics and absorbers"); panel(ax, "b")
ax.legend(fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
ax = axs[2]
ax.plot(E, phi / phi.max(), color="0.5", label="per 0.1 % BW")
ax.plot(E, dens / dens.max(), color=C["accent"], label="per keV")
ax.plot(E, dens_full / dens_full.max(), ":", color=C["accent"], lw=0.9, label="per keV, G$_1$ source")
for z in (31, 33):
    ax.axvline(xraylib.EdgeEnergy(z, xraylib.K_SHELL), color="0.6", lw=0.6, ls=":")
ax.text(12.5, 0.06, "Ga, As K edges", fontsize=6, color="0.4", va="bottom", ha="left")
ax.set_xlim(3, 40); ax.set_ylim(0, 1.6)
ax.set_xlabel("photon energy (keV)"); ax.set_ylabel("normalised")
ax.set_title("Bandwidth Jacobian"); ax.legend(loc="upper center", fontsize=5.6, borderaxespad=0.2); panel(ax, "c")
save(fig, "fig03_beam")

fig, axs = plt.subplots(1, 3, figsize=(TEXTW, 2.3), gridspec_kw={"wspace": 0.42, "width_ratios": [1.15, 1, 1]})
ax = axs[0]
L_ = XB.BM32_LAYOUT
zpos = {"s1": L_["s1"], lab_m1: L_["m1"], lab_m2: L_["m2"], "s2": L_["s2"], "s3": L_["s3"], lab_kbv: L_["kbv"]}
short = lambda k: k.split(" Ir")[0]
for k, z in zpos.items():
    v = INFO["vertical_limits_urad"].get(k)
    if v is not None:
        ax.semilogy([z, z], [0.5, v], color=C["accent"], lw=2.2, solid_capstyle="butt")
        dz = {"m1": -0.9, "m2": 0.9, "s3": -1.0, "KB-V": 0.6}.get(short(k), 0.0)
        ax.text(z + dz, v * 1.25, short(k), ha="center", va="bottom", fontsize=5.8, color=C["accent"])
for k, v in INFO["horizontal_limits_urad"].items():
    z = {"s1": L_["s1"], "s2": L_["s2"], "s3": L_["s3"]}.get(k, L_["kbh"])
    ax.semilogy([z + 0.45], [v], "s", ms=3, color=C["Ga"])
ax.axhline(fwhm_psi["20"], color="0.4", lw=0.7, ls="--")
ax.text(37.0, fwhm_psi["20"] * 1.12, "FWHM, 20 keV", fontsize=5.6, color="0.3")
ax.axhline(1e6 / src.gamma, color="0.6", lw=0.6, ls=":")
ax.text(37.0, 1e6 / src.gamma * 0.62, "1/γ", fontsize=5.6, color="0.4")
ax.plot([], [], color=C["accent"], lw=2, label="vertical"); ax.plot([], [], "s", ms=3, color=C["Ga"], label="horizontal")
ax.set_xlim(25, 46); ax.set_ylim(1, 1500); ax.set_xlabel("distance from source (m)")
ax.set_ylabel("acceptance at source (µrad)"); ax.set_title("Apertures")
ax.legend(loc="lower left", fontsize=5.6, bbox_to_anchor=(0.47, 0.0)); panel(ax, "a")
ax = axs[1]
for e0, col in ((5, C["In"]), (10, C["As"]), (20, C["Ga"]), (30, C["model"])):
    w = ADEN[np.argmin(abs(E - e0))]
    ax.plot(PSI, w / w.max(), color=col, label=f"{e0} keV")
a3 = INFO["vertical_acceptance_urad"]
ax.axvspan(-a3 / 2, a3 / 2, color="0.75", lw=0, label=f"s3: {a3:.1f} µrad")
ax.set_xlim(-300, 300); ax.set_ylim(0, 1.5); ax.set_xlabel("vertical angle ψ (µrad)"); ax.set_ylabel("angular density (rel.)")
ax.set_title("Vertical emission"); ax.legend(loc="upper center", fontsize=5.6, ncol=3, columnspacing=0.8, handlelength=1.2); panel(ax, "b")
ax = axs[2]
k0 = R["beam"]["k_gaas_full"]
for o, ls in zip(OFFS, ("-", "--", ":")):
    ax.semilogx(ACC, 100 * (np.array(R["beam"]["acc_scan"]["k"][str(o)]) / k0 - 1), ls, marker="o", ms=2.5,
                color=C["Ga"], label=f"offset {o:.0f} µrad")
ax.axvline(a3, color="0.5", lw=0.6, ls="--"); ax.axhline(0, color="0.7", lw=0.5)
ax.set_xlabel("vertical acceptance (µrad)"); ax.set_ylabel("change of Ga/As sensitivity (%)")
ax.set_title("Effect on Ga/As"); ax.legend(loc="lower left", fontsize=5.8); panel(ax, "c")
save(fig, "fig03b_acceptance")
print(f"beam ({time.time()-T0:.0f}s)")

# ----------------------------------------------------------------------------- detector
Eev = np.linspace(1000, 20000, 800)
rho_air = XB.air_density()
t_air = np.array([np.exp(-xraylib.CS_Total_CP(XB.AIR_FORMULA, e / 1e3) * rho_air * 20.0) for e in Eev])
t_win = np.array([np.exp(-xraylib.CS_Total_CP("C", e / 1e3) * 2.2 * 1e-4) for e in Eev])
absorb = np.array([1 - np.exp(-xraylib.CS_Photo(14, e / 1e3) * 2.33 * 450e-4) for e in Eev])
eff = DET.efficiency(Eev)
esc = np.array([sum(f for _, f in DET.escape_lines(e)) for e in Eev])
R["detector"] = {k: float(DET.efficiency(v * 1000)) for k, v in LINE_E.items()}
d21 = dataclasses.replace(DET, air_path_cm=21.0)
R["detector"]["In_La1_per_cm"] = float(d21.efficiency(LINE_E["In La1"] * 1000) / DET.efficiency(LINE_E["In La1"] * 1000))
R["detector"]["Ga_Ka1_per_cm"] = float(d21.efficiency(LINE_E["Ga Ka1"] * 1000) / DET.efficiency(LINE_E["Ga Ka1"] * 1000))
d990 = dataclasses.replace(DET, air_pressure_hpa=990.0)
R["detector"]["In_La1_990hPa"] = float(d990.efficiency(LINE_E["In La1"] * 1000) / DET.efficiency(LINE_E["In La1"] * 1000))
R["detector"]["escape_Ga_Ka1"] = float(sum(f for _, f in DET.escape_lines(LINE_E["Ga Ka1"] * 1000)))
R["detector"]["escape_In_La1"] = float(sum(f for _, f in DET.escape_lines(LINE_E["In La1"] * 1000)))
R["detector"]["escape_max"] = float(esc.max()); R["detector"]["escape_max_ev"] = float(Eev[np.argmax(esc)])

fig, axs = plt.subplots(1, 3, figsize=(TEXTW, 2.1), gridspec_kw={"wspace": 0.45})
ax = axs[0]
ax.plot(Eev / 1e3, t_air, color="0.55", label="20 cm air")
ax.plot(Eev / 1e3, t_win, color="0.75", ls="--", label="1 µm C window")
ax.plot(Eev / 1e3, absorb, color=C["Ga"], label="450 µm Si (photo)")
ax.plot(Eev / 1e3, eff, color="k", lw=1.4, label="total")
for k, col in (("In La1", C["In"]), ("Ga Ka1", C["Ga"]), ("As Ka1", C["As"])):
    ax.axvline(LINE_E[k], color=col, lw=0.6, ls=":")
ax.set_xlabel("photon energy (keV)"); ax.set_ylabel("probability"); ax.set_ylim(0, 1.05)
ax.set_title("Detection efficiency"); ax.legend(loc="lower right", fontsize=6); panel(ax, "a")
ax = axs[1]
gaps = np.linspace(10, 30, 41)
for k, col in (("In La1", C["In"]), ("Ga Ka1", C["Ga"]), ("As Ka1", C["As"])):
    ax.semilogy(gaps, [dataclasses.replace(DET, air_path_cm=g).efficiency(LINE_E[k] * 1000) for g in gaps],
                color=col, label=greek(k))
ax.axvline(20, color="0.6", lw=0.6, ls=":")
ax.set_xlabel("sample–detector air path (cm)"); ax.set_ylabel("efficiency")
ax.set_title("Air-gap sensitivity"); ax.legend(loc="lower left"); panel(ax, "b")
ax = axs[2]
ax.plot(Eev / 1e3, 100 * esc, color=C["esc"])
ax.set_xlabel("photon energy (keV)"); ax.set_ylabel("Si escape fraction (%)")
ax.set_title("Escape probability"); ax.set_xlim(1, 20); panel(ax, "c")
save(fig, "fig04_detector")
print(f"detector ({time.time()-T0:.0f}s)")

# ----------------------------------------------------------------------------- matrix: thickness dependence
thick = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.9, 3.0, 5.0, 10.0])
keys3 = ("Ga Ka1", "As Ka1", "In La1")
abs_c, sf_c = {k: [] for k in keys3}, {k: [] for k in keys3}
for t in thick:
    mm = XM.MatrixModel(XM.Layer("In0.5Ga0.5As", 5.5, t), GEOM, E, phi)
    for k in keys3:
        z, mac = LINES[k]
        abs_c[k].append(mm.absorption_factor(z, mac, LINE_E[k]))
        sf_c[k].append(mm.enhancement(z, mac, LINE_E[k]))
R["matrix_curves"] = dict(thickness_um=thick.tolist(),
                          absorption={k: list(map(float, v)) for k, v in abs_c.items()},
                          enhancement={k: list(map(float, v)) for k, v in sf_c.items()})
print(f"matrix curves ({time.time()-T0:.0f}s)")

# ----------------------------------------------------------------------------- line groups by physics level
mm0 = XM.MatrixModel(STACK, GEOM, E, phi, target=TARGET)
kw = dict(families=FAMILIES, min_rel=MIN_REL)
LEVELS = {
    "scalar": XF.element_line_groups(ELEMENTS, *ROI_EV, e0_eff_kev=20.0, **kw),
    "white": XF.element_line_groups(ELEMENTS, *ROI_EV, incident_spectrum=(E, phi), **kw),
    "white+eff": XF.element_line_groups(ELEMENTS, *ROI_EV, incident_spectrum=(E, phi), detector=DET, **kw),
    "matrix": XF.element_line_groups(ELEMENTS, *ROI_EV, detector=DET, matrix=mm0, **kw),
}
print(f"line groups ({time.time()-T0:.0f}s)")
R["In_L_rel"], R["As_K_rel"] = {}, {}
for lvl, gl in LEVELS.items():
    gi = {l.name: float(l.rel) for l in next(x for x in gl if x.label == "In L").lines}
    ga = {l.name: float(l.rel) for l in next(x for x in gl if x.label == "As K").lines}
    R["In_L_rel"][lvl] = {k: gi.get(k) for k in ("L3M5", "L2M4", "L3N5", "L1M3")}
    R["As_K_rel"][lvl] = {k: ga.get(k) for k in ("KL2", "KM3")}

# ----------------------------------------------------------------------------- pile-up
P = XF.pileup_spectra(spectra, CAL, rt)
TMPL = XF.average_pileup_template(P, scale)
g_pp = XF.element_line_groups(ELEMENTS, *PILEUP_ROI_EV, incident_spectrum=(E, phi), detector=DET, **kw)
r_pp = XF.fit_average_spectrum(avg, g_pp, CAL, DET, ROI_PP, yerr=cnt_err, background_order=BKG + 2,
                               escape=True, pileup_template=TMPL)
r_pp0 = XF.fit_average_spectrum(avg, g_pp, CAL, DET, ROI_PP, yerr=cnt_err, background_order=BKG + 2,
                                escape=True)
TAU = float(r_pp.model.pileup_tau_us(r_pp.popt))
R["pileup"] = dict(tau_us=TAU, tau_err_us=float(r_pp.perr[-1]), chi2_with=float(r_pp.chi2_red),
                   chi2_without=float(r_pp0.chi2_red),
                   pair_probability=float(TAU * 1e-6 * icr.mean()))
FKW = dict(yerr=cnt_err, background_order=BKG, escape=True, pileup_template=TMPL,
           fit_pileup=False, pileup_tau_us=TAU)
print(f"pile-up ({time.time()-T0:.0f}s)")

# ----------------------------------------------------------------------------- fits per level
R["levels"] = {}
for lvl, gl in LEVELS.items():
    r = XF.fit_average_spectrum(avg, gl, CAL, DET, ROI, **FKW)
    R["levels"][lvl] = dict(chi2=float(r.chi2_red), In=float(r.areas["In L"]),
                            Ga=float(r.areas["Ga K"]), As=float(r.areas["As K"]))
r_noesc = XF.fit_average_spectrum(avg, LEVELS["matrix"], CAL, DET, ROI, **{**FKW, "escape": False})
R["escape_test"] = dict(Ni_without=float(r_noesc.areas["Ni K"]), chi2_without=float(r_noesc.chi2_red))
print(f"level fits ({time.time()-T0:.0f}s)")

# ----------------------------------------------------------------------------- self-consistent quantification
groups = LEVELS["matrix"]
res = XF.fit_average_spectrum(avg, groups, CAL, DET, ROI, **FKW)
Q = XM.quantify_layer(res.areas, {g.label: g for g in groups}, mm0, DET, REF, ANALYTES)
groups = XF.element_line_groups(ELEMENTS, *ROI_EV, detector=DET, matrix=Q["matrix"], **kw)
res = XF.fit_average_spectrum(avg, groups, CAL, DET, ROI, **FKW)
Q = XM.quantify_layer(res.areas, {g.label: g for g in groups}, Q["matrix"], DET, REF, ANALYTES)
GQ = {g.label: g for g in groups}
GIDX = {g.label: i for i, g in enumerate(groups)}
R["escape_test"]["Ni_with"] = float(res.areas["Ni K"])
R["escape_test"]["chi2_with"] = float(res.chi2_red)
pp = res.params()
R["fit"] = dict(chi2=float(res.chi2_red), n_channels=int(res.channels.size), n_free=int(len(res.popt)),
                n_lines=int(sum(len(g.lines) for g in groups)), n_groups=len(groups),
                noise_ev=pp["noise_ev"][0], fano=pp["fano"][0], gamma_ev=pp["gamma_ev"][0],
                fwhm_mnka_ev=float(DET.fwhm_ev(5899, noise_ev=pp["noise_ev"][0], fano=pp["fano"][0])),
                fwhm_gaka_ev=float(DET.fwhm_ev(LINE_E["Ga Ka1"] * 1000, noise_ev=pp["noise_ev"][0],
                                               fano=pp["fano"][0])),
                areas={k: float(v) for k, v in res.areas.items()},
                area_errors={k: float(v) for k, v in res.area_errors.items()},
                resid_rms_pct=float(100 * np.sqrt(((res.residual / res.y.max()) ** 2).mean())))
R["quant"] = dict(ratios={k: float(v) for k, v in Q["ratios"].items()},
                  k={k: float(v) for k, v in Q["k"].items()},
                  enhancement={k: float(v) for k, v in Q["enhancement"].items()},
                  absorption={k: float(v) for k, v in Q["absorption"].items()},
                  other_layers={k: float(v) for k, v in Q["other_layers"].items()},
                  iterations=int(Q["iterations"]),
                  closure=float(Q["ratios"]["Ga"] + Q["ratios"]["In"]),
                  layer=Q["layer"].formula)
mmc = Q["matrix"]
e_in = LINE_E["In La1"]
R["quant"]["cap_In_SF_over_primary"] = float(mmc.secondary(49, xraylib.L3M5_LINE, e_in, layer=0)
                                             / mmc.primary(49, xraylib.L3M5_LINE, e_in, layer=0))
R["quant"]["cap_transmission"] = {k: float(np.exp(-STACK[0].mu(LINE_E[k]) / GEOM.sin_out * STACK[0].mass_thickness))
                                  for k in ("In La1", "Ga Ka1", "As Ka1")}
sf_parts = mmc.secondary(31, xraylib.KL3_LINE, LINE_E["Ga Ka1"], by_exciter=True)[1]
p_ga = mmc.primary(31, xraylib.KL3_LINE, LINE_E["Ga Ka1"])
R["quant"]["Ga_SF_by_exciter"] = {k: float(v / p_ga) for k, v in sorted(sf_parts.items(), key=lambda kv: -kv[1])[:4]}
sf_in = mmc.secondary(49, xraylib.L3M5_LINE, e_in, by_exciter=True)[1]
p_in = mmc.primary(49, xraylib.L3M5_LINE, e_in)
R["quant"]["In_SF_by_exciter"] = {k: float(v / p_in) for k, v in sorted(sf_in.items(), key=lambda kv: -kv[1])[:4]}
print(f"quantification ({time.time()-T0:.0f}s)")

mm_nc = XM.MatrixModel(XM.Stack([STACK[1], STACK[2]]), GEOM, E, phi, target=0)
R["quant"]["no_cap"] = {k: float(v) for k, v in XM.quantify_layer(res.areas, GQ, mm_nc, DET, REF, ANALYTES)["ratios"].items()}
gl = {g.label: g for g in LEVELS["white+eff"]}
rr = XF.fit_average_spectrum(avg, LEVELS["white+eff"], CAL, DET, ROI, **FKW)
R["quant"]["no_matrix"] = {el: float(XF.number_ratio(rr.areas[l], rr.areas[REF], gl[l], gl[REF])[0])
                           for el, l in (("Ga", "Ga K"), ("In", "In L"))}


def requant(detector=DET, geometry=GEOM, stack=None, target=TARGET, spectrum=None):
    mm = XM.MatrixModel(stack or Q["matrix"].stack, geometry, E, phi if spectrum is None else spectrum,
                        target=target)
    return {k: float(v) for k, v in XM.quantify_layer(res.areas, GQ, mm, detector, REF, ANALYTES)["ratios"].items()}


ST = Q["matrix"].stack
cases = [("nominal", {}),
         ("air 18 cm", dict(detector=dataclasses.replace(DET, air_path_cm=18.0))),
         ("air 22 cm", dict(detector=dataclasses.replace(DET, air_path_cm=22.0))),
         ("take-off 30°", dict(geometry=XM.Geometry(40, 30))),
         ("take-off 50°", dict(geometry=XM.Geometry(40, 50))),
         ("film 1.52 µm", dict(stack=ST.replace(1, ST[1].with_thickness(1.52)))),
         ("film 2.28 µm", dict(stack=ST.replace(1, ST[1].with_thickness(2.28)))),
         ("cap 30 nm", dict(stack=ST.replace(0, ST[0].with_thickness(0.03)))),
         ("cap 70 nm", dict(stack=ST.replace(0, ST[0].with_thickness(0.07)))),
         ("no cap", dict(stack=XM.Stack([ST[1], ST[2]]), target=0)),
         ("990 hPa", dict(detector=dataclasses.replace(DET, air_pressure_hpa=990.0))),
         ("source G1 (all ψ)", dict(spectrum=phi_full)),
         ("beam offset 25 µrad", dict(spectrum=XB.bm32_from_h5(H5, vertical_offset_urad=25.0).spectrum(E)[1])),
         ("KB angles 2.5 mrad", dict(spectrum=XB.bm32_from_h5(H5, kbv_angle_mrad=2.5, kbh_angle_mrad=2.5).spectrum(E)[1])),
         ("KB angles 3.5 mrad", dict(spectrum=XB.bm32_from_h5(H5, kbv_angle_mrad=3.5, kbh_angle_mrad=3.5).spectrum(E)[1]))]
R["sensitivity"] = {lab: requant(**kwc) for lab, kwc in cases}
from scipy.optimize import brentq
_clos = lambda x: (lambda r: r["Ga"] + r["In"] - 1.0)(requant(detector=dataclasses.replace(DET, air_path_cm=x)))
R["air_for_closure_cm"] = float(brentq(_clos, 12.0, 20.0, xtol=0.01))
R["cap_scan"] = {}
for tn in (0, 100, 200, 300, 400, 500):
    st = XM.Stack([ST[1], ST[2]]) if tn == 0 else ST.replace(0, ST[0].with_thickness(tn / 1000))
    R["cap_scan"][str(tn)] = requant(stack=st, target=0 if tn == 0 else 1)
print(f"sensitivity ({time.time()-T0:.0f}s)")

# ----------------------------------------------------------------------------- resolution tracking + maps
WIN = XF.width_window(groups, res.areas, DET)
PROF = XF.peak_width_profile(spectra, CAL, WIN[:2], bin_size=n_fast)
STEPS = XF.detect_steps(PROF["width_ev"], window=5, threshold=6.0, min_step=1.0)
CSTEPS = XF.detect_steps(PROF["centroid_ev"], window=5, threshold=6.0, min_step=1.0)
SEG = XF.segment_labels(STEPS, n_slow, n_fast, n_pix)
SEGF = XF.fit_segments(corrected, spectra, scale, SEG, groups, CAL, DET, ROI, pileup=P, **FKW)
SH = {lab: r.popt for lab, r in SEGF.items()}
MAPS = {}
for name, kwm in {"none": dict(), "segments": dict(segments=SEG), "linear": dict(width_correction=True),
                  "both": dict(segments=SEG, width_correction=True)}.items():
    MAPS[name] = XF.fit_map(spectra, scale, res.model, SH if "segments" in kwm else res.popt,
                            scan.scan_shape, pileup=P, **kwm)
m_raw = XF.fit_map(spectra, np.ones_like(scale), res.model, res.popt, scan.scan_shape, pileup=P)
inv = np.nanmax(np.abs((MAPS["none"].area_map("Ga K") / MAPS["none"].area_map("As K"))
                       / (m_raw.area_map("Ga K") / m_raw.area_map("As K")) - 1))
rowmed = {k: np.nanmedian(v.area_map("Ga K") / v.area_map("As K"), axis=1) for k, v in MAPS.items()}
s0 = STEPS[0] if STEPS else n_slow // 3
wc = MAPS["both"].width_change_map()
k_change = int(np.argmax(wc[s0] > -0.025))


def roi_counts(lo, hi):
    msk = (Ech >= lo) & (Ech <= hi)
    return spectra[:, msk].sum(1).reshape(scan.scan_shape).sum(1)


roi_ratio = roi_counts(8950, 9550) / roi_counts(10250, 10850)
R["resolution"] = dict(
    window_ev=[float(WIN[0]), float(WIN[1])], window_line=WIN[2], steps=list(map(int, STEPS)),
    centroid_steps=list(map(int, CSTEPS)), width_err_ev=float(np.nanmedian(PROF["width_err_ev"])),
    width_before=float(np.nanmean(PROF["width_ev"][:s0])), width_after=float(np.nanmean(PROF["width_ev"][s0:])),
    centroid_range_ev=float(np.nanmax(PROF["centroid_ev"]) - np.nanmin(PROF["centroid_ev"])),
    fwhm_segments={str(l): float(DET.fwhm_ev(LINE_E["Ga Ka1"] * 1000, noise_ev=r.params()["noise_ev"][0],
                                             fano=r.params()["fano"][0])) for l, r in SEGF.items()},
    gamma_segments={str(l): float(r.params()["gamma_ev"][0]) for l, r in SEGF.items()},
    chi2_segments={str(l): float(r.chi2_red) for l, r in SEGF.items()},
    invariance=float(inv), change_pixel_in_row=k_change,
    change_time_s=float(epoch.reshape(scan.scan_shape)[s0, k_change] - epoch[0]),
    width_change_first_pixels_pct=[float(100 * x) for x in wc[s0, :8]],
    roi_sum_ratio_before_after=[float(roi_ratio[:s0].mean()), float(roi_ratio[s0:].mean())],
    icr_before_after_kcps=[float(icr.reshape(scan.scan_shape)[:s0].mean() / 1e3),
                           float(icr.reshape(scan.scan_shape)[s0:].mean() / 1e3)],
    modes={})
for name, mres in MAPS.items():
    rat = mres.area_map("Ga K") / mres.area_map("As K")
    rm = rowmed[name]
    R["resolution"]["modes"][name] = dict(
        step_pct=float(100 * (np.nanmean(rm[:s0]) / np.nanmean(rm[s0:]) - 1)),
        spread_pct=float(100 * np.nanstd(rat / np.nanmedian(rat))),
        chi2_median=float(np.nanmedian(mres.chi2_red)),
        chi2_p5_p95=[float(np.nanpercentile(mres.chi2_red, 5)), float(np.nanpercentile(mres.chi2_red, 95))])
print(f"resolution + maps ({time.time()-T0:.0f}s)")

MB = MAPS["both"]
XMAP, DXMAP = XM.atom_ratio(Q, "Ga K", MB.area_map("Ga K"), MB.area_map("As K"),
                            MB.error_map("Ga K"), MB.error_map("As K"))
XN, DXN = XM.atom_ratio(Q, "Ga K", MAPS["none"].area_map("Ga K"), MAPS["none"].area_map("As K"),
                        MAPS["none"].error_map("Ga K"), MAPS["none"].error_map("As K"))
v, vn = XMAP[np.isfinite(XMAP)], XN[np.isfinite(XN)]
sig = float(np.nanmedian(DXMAP) * np.sqrt(np.nanmedian(MB.chi2_red)))
sig_n = float(np.nanmedian(DXN) * np.sqrt(np.nanmedian(MAPS["none"].chi2_red)))
R["maps"] = dict(x_median=float(np.median(v)), x_spread=float(v.std()), x_stat=sig,
                 x_excess=float(np.sqrt(max(v.std() ** 2 - sig ** 2, 0))),
                 x_none_median=float(np.median(vn)), x_none_spread=float(vn.std()),
                 x_excess_none=float(np.sqrt(max(vn.std() ** 2 - sig_n ** 2, 0))),
                 snr={lab: float(np.nanmedian(MB.snr(lab))) for lab in MB.group_labels})

# --- storage-ring refills: residual steps of the monitor-normalised areas
jumps = np.where(np.diff(mon.astype(float)) > 0.05 * mon.mean())[0] + 1
ga_flat = MB.area_map("Ga K", total=True).ravel()
as_flat = MB.area_map("As K", total=True).ravel()
rat_flat = XMAP.ravel()
R["refills"] = []
for j in jumps:
    b, a = slice(max(j - 200, 0), j), slice(j, min(j + 200, n_pix))
    R["refills"].append(dict(
        pixel=int(j), row=int(j // n_fast), col=int(j % n_fast), time_s=float(epoch[j] - epoch[0]),
        monitor_jump_pct=float(100 * (mon[a].mean() / mon[b].mean() - 1)),
        raw_rate_jump_pct=float(100 * ((spectra[a].sum(1) / lt[a]).mean() / (spectra[b].sum(1) / lt[b]).mean() - 1)),
        ga_area_step_pct=float(100 * (np.nanmean(ga_flat[a]) / np.nanmean(ga_flat[b]) - 1)),
        as_area_step_pct=float(100 * (np.nanmean(as_flat[a]) / np.nanmean(as_flat[b]) - 1)),
        ratio_step_pct=float(100 * (np.nanmean(rat_flat[a]) / np.nanmean(rat_flat[b]) - 1))))
rows_ga = np.nanmedian(MB.area_map("Ga K", total=True), axis=1)
R["data"]["ga_area_row_spread_pct"] = float(100 * np.nanstd(rows_ga) / np.nanmean(rows_ga))
R["data"]["rate_over_monitor_row_spread_pct"] = float(100 * np.std(
    (spectra.sum(1) / lt / mon).reshape(scan.scan_shape).mean(1)) / (spectra.sum(1) / lt / mon).mean())
R["data"]["rate_row_spread_pct"] = float(100 * np.std((spectra.sum(1) / lt).reshape(scan.scan_shape).mean(1))
                                         / (spectra.sum(1) / lt).mean())
R["data"]["rate_monitor_corr"] = float(np.corrcoef((spectra.sum(1) / lt).reshape(scan.scan_shape).mean(1),
                                                   mon.reshape(scan.scan_shape).mean(1))[0, 1])

# ============================================================================= remaining figures
# --- fig 1: workflow
fig, ax = plt.subplots(figsize=(TEXTW, 3.4)); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 60)


def box(x, y, w, h, txt, fc="#f4f6f7"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="0.3", lw=0.8))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=6.3, linespacing=1.25)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.3"))


main = [(52, "Raw MCA spectra (BLISS HDF5)\none spectrum per pixel"),
        (44, "Normalisation\nmonitor × live time"),
        (36, "Map-averaged spectrum\nand counting error"),
        (28, "Constrained fit of the\naverage spectrum"),
        (20, "Self-consistent composition\nof the film (iterated)"),
        (12, "Resolution tracking\nalong the scan"),
        (4, "Per-pixel linear solve\n→ atom-ratio maps")]
for y, t in main:
    box(36, y, 28, 6, t, fc="#eaf2f8" if ("fit" in t or "composition" in t) else "#f4f6f7")
for (y0, _), (y1, _) in zip(main[:-1], main[1:]):
    arrow(50, y0 - 0.6, 50, y1 + 6.6)
for y, t, tgt in ((40, "Beam model\nbending magnet, slit acceptance,\nIr mirrors, Be, 3.3 m air\n→ spectral density dΦ/dE", 32.5),
                  (21, "Fundamental parameters\nKissel cross sections,\nfluorescence yields,\nCoster–Kronig transitions", 29.5)):
    box(0.5, y, 28, 11, t, fc="#fdf2e9"); arrow(29.3, y + 5.5, 35.4, tgt)
for y, t, tgt in ((40, "Detector model\nFano resolution, Voigt shape,\nSi escape, pile-up,\nefficiency (air, C, Si)", 32.5),
                  (21, "Sample model\nInP / (In,Ga)As / Si stack,\nabsorption, secondary\nfluorescence, take-off angle", 29.5)):
    box(71.5, y, 28, 11, t, fc="#fdf2e9"); arrow(70.7, y + 5.5, 64.6, tgt)
arrow(70.7, 24.0, 64.6, 23.0)
save(fig, "fig01_workflow")

# --- fig 2: data overview
fig = plt.figure(figsize=(TEXTW, 5.6))
gs = fig.add_gridspec(3, 3, height_ratios=[1.3, 1, 0.8], hspace=0.7, wspace=0.38)
ax = fig.add_subplot(gs[0, :])
ax.semilogy(Ech / 1e3, avg, color=C["data"], lw=0.7)
ax.axvspan(ROI_EV[0] / 1e3, ROI_EV[1] / 1e3, color="0.92", zorder=0)
for nm, e, col in (("Ar K", 2.96, C["Ar"]), ("In L", 3.29, C["In"]), ("Ca K", 3.69, C["Ca"]),
                   ("Ga esc. / Ni", 7.48, C["Ni"]), ("Cu Kα", 8.05, C["Cu"]), ("Ga Kα", 9.25, C["Ga"]),
                   ("As Kα", 10.54, C["As"]), ("As Kβ", 11.73, C["As"]), ("2×Ga Kα", 18.50, C["pile"]),
                   ("Ga+As Kα", 19.79, C["pile"])):
    i = np.argmin(abs(Ech / 1e3 - e))
    ax.annotate(nm, (e, avg[i] * 1.4), fontsize=6, ha="center", va="bottom", color=col, rotation=90)
ax.set_xlim(0, Ech[-1] / 1e3); ax.set_ylim(3e-2, avg.max() * 40)
ax.set_xlabel("energy (keV)"); ax.set_ylabel("counts / live s (norm.)")
ax.set_title(f"Map-averaged spectrum, {n_pix} pixels (grey band: fit region)"); panel(ax, "a")
for j, (dat, ttl) in enumerate(((mon, "Monitor (counts)"), (lt, "Live time (s)"), (fdt, "Dead-time fraction"))):
    ax = fig.add_subplot(gs[1, j])
    im = ax.imshow(dat.reshape(scan.scan_shape), origin="lower", cmap="viridis", aspect="auto")
    ax.set_title(ttl); ax.set_xlabel("fast (pixel)")
    if j == 0:
        ax.set_ylabel("slow (pixel)")
    cb = fig.colorbar(im, ax=ax, fraction=0.05); cb.ax.tick_params(labelsize=6)
    panel(ax, "bcd"[j])
ax = fig.add_subplot(gs[2, :])
rowm = lambda q: np.nanmean(np.asarray(q, float).reshape(scan.scan_shape), axis=1)
rr = rowm(spectra.sum(1) / lt)
ax.plot(rowm(mon) / np.mean(mon), color="0.55", lw=0.9, label="monitor")
ax.plot(rr / rr.mean(), color="k", lw=0.9, label="total count rate (per live s)")
ga_r = np.nanmedian(MB.area_map("Ga K", total=True), axis=1)
ax.plot(ga_r / np.nanmean(ga_r), color=C["Ga"], lw=1.1, label="Ga K area after normalisation")
for rf in R["refills"]:
    ax.axvline(rf["row"] + rf["col"] / n_fast, color=C["model"], lw=0.6, ls="--")
ax.set_xlabel("map row (acquisition order)"); ax.set_ylabel("relative")
ax.set_title("Storage-ring refills (dashed)"); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=6)
ax.set_ylim(0.86, 1.16); panel(ax, "e")
save(fig, "fig02_data")

# --- fig 5: matrix
fig, axs = plt.subplots(1, 3, figsize=(TEXTW, 2.2), gridspec_kw={"wspace": 0.5, "width_ratios": [0.85, 1, 1]})
ax = axs[0]; ax.set_axis_off(); ax.set_xlim(0, 10); ax.set_ylim(0, 13)
for y, h, col, lab in ((8.6, 0.7, "#d6eaf8", "InP cap, 50 nm"), (4.2, 4.4, "#fdebd0", "(In,Ga)As, 1.9 µm"),
                       (0.3, 3.9, "#e5e8e8", "Si substrate (bulk)")):
    ax.add_patch(plt.Rectangle((0.5, y), 9, h, fc=col, ec="0.4", lw=0.6))
    ax.text(5, y + h / 2, lab, ha="center", va="center", fontsize=6.5)
ax.annotate("", xy=(4.6, 9.4), xytext=(0.8, 12.6), arrowprops=dict(arrowstyle="-|>", color=C["model"], lw=1))
ax.annotate("", xy=(9.4, 12.6), xytext=(5.4, 9.4), arrowprops=dict(arrowstyle="-|>", color=C["accent"], lw=1))
ax.text(0.6, 10.6, "beam\n$\\psi_{in}$", fontsize=6, color=C["model"])
ax.text(8.0, 10.6, "SDD\n$\\psi_{out}$", fontsize=6, color=C["accent"])
ax.set_title("Sample model"); panel(ax, "a")
for ax, dat, ttl, lab, let in ((axs[1], abs_c, "Primary absorption", "absorption factor $A$", "b"),
                               (axs[2], sf_c, "Secondary fluorescence", "$S/P$ (%)", "c")):
    for k, col in (("In La1", C["In"]), ("Ga Ka1", C["Ga"]), ("As Ka1", C["As"])):
        y = np.array(dat[k]) * (100 if let == "c" else 1)
        ax.semilogx(thick, y, "o-", ms=2.5, color=col, label=greek(k))
    ax.axvline(1.9, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("(In,Ga)As thickness (µm)"); ax.set_ylabel(lab); ax.set_title(ttl); panel(ax, let)
axs[1].legend(loc="lower left")
save(fig, "fig05_matrix")

# --- fig 6: average-spectrum fit
model = res.model
x = CAL.energy(res.channels) / 1e3
fig, (ax, axr) = plt.subplots(2, 1, figsize=(TEXTW, 3.9), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
ax.semilogy(x, res.y, color=C["data"], lw=0.8, label="data")
ax.semilogy(x, res.yfit, color=C["model"], lw=0.9, label="model")
ax.semilogy(x, model.background(res.popt), color=C["bkg"], lw=0.7, ls="--", label="background")
for i, g in enumerate(model.groups):
    ax.semilogy(x, model.group_profile(i, res.popt) + 1e-9, lw=0.8, color=C.get(g.element, "0.5"), label=g.label)
ax.semilogy(x, model.pileup_profile(res.popt) + 1e-9, lw=0.8, color=C["pile"], label="pile-up")
ax.set_ylim(0.05, res.y.max() * 3); ax.set_ylabel("counts / live s (norm.)")
ax.legend(ncol=6, loc="upper left", fontsize=6)
axr.plot(x, res.residual_normalised, color="0.2", lw=0.6)
axr.axhline(0, color="0.6", lw=0.5); axr.set_ylabel("residual / σ"); axr.set_xlabel("energy (keV)")
save(fig, "fig06_fit")

# --- fig 7: escape and pile-up
m_noesc = XF.XRFModel(model.groups, CAL, DET, res.channels, background_order=BKG, escape=False,
                      pileup_template=TMPL, fit_pileup=False, pileup_tau_us=TAU)
fig, axs = plt.subplots(1, 2, figsize=(TEXTW, 2.3), gridspec_kw={"wspace": 0.3})
ax = axs[0]
sel = (x > 6.9) & (x < 9.1)
ax.plot(x[sel], res.y[sel], color=C["data"], lw=0.8, label="data")
ax.plot(x[sel], res.yfit[sel], color=C["model"], lw=0.9, label="model")
for lab, col in (("Ga K", C["Ga"]), ("As K", C["As"])):
    i = GIDX[lab]
    e_comp = model.group_profile(i, res.popt) - m_noesc.group_profile(i, res.popt)
    ax.fill_between(x[sel], 0, e_comp[sel], color=col, alpha=0.5, lw=0, label=f"{lab} escape")
ax.plot(x[sel], model.group_profile(GIDX["Ni K"], res.popt)[sel], color=C["Ni"], lw=0.8, label="Ni K (fitted)")
sel2 = (x > 6.9) & (x < 8.7)
ax.set_xlim(6.9, 8.7); ax.set_ylim(0, 1.25 * res.y[sel2].max())
ax.set_xlabel("energy (keV)"); ax.set_ylabel("counts / live s (norm.)")
ax.set_title("Si escape peaks"); ax.legend(fontsize=6, loc="upper left"); panel(ax, "a")
ax = axs[1]
xp = CAL.energy(r_pp.channels) / 1e3
sp = xp > 12.8
ax.plot(xp[sp], r_pp.y[sp], color=C["data"], lw=0.8, label="data")
ax.plot(xp[sp], r_pp.yfit[sp], color=C["model"], lw=0.9, label="model")
ax.plot(xp[sp], r_pp0.yfit[sp], color="0.5", lw=0.8, ls="--", label="model without pile-up")
ax.fill_between(xp[sp], 0, r_pp.model.pileup_profile(r_pp.popt)[sp], color=C["pile"], alpha=0.6, lw=0,
                label=f"pile-up (τ = {TAU*1e3:.0f} ns)")
ax.set_xlabel("energy (keV)"); ax.set_ylim(0, None); ax.set_title("Pile-up above the fit region")
ax.legend(fontsize=6, loc="upper left"); panel(ax, "b")
save(fig, "fig07_escape_pileup")

# --- fig 8: resolution tracking
rows = np.arange(n_slow)
fig = plt.figure(figsize=(TEXTW, 4.2))
gs = fig.add_gridspec(3, 2, width_ratios=[1.6, 1], hspace=0.15, wspace=0.35)
ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[1, 0], sharex=ax1); ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
ax1.errorbar(rows, PROF["width_ev"], yerr=PROF["width_err_ev"], fmt="o-", ms=2, lw=0.7, color="0.2")
ax1.set_ylabel(f"{WIN[2]} rms width (eV)"); panel(ax1, "a")
ax2.plot(rows, PROF["centroid_ev"], "o-", ms=2, lw=0.7, color="0.45"); ax2.set_ylabel("centroid (eV)")
for name, col, ls in (("none", C["model"], "-"), ("segments", C["accent"], "-"), ("linear", C["Ga"], "--"),
                      ("both", "k", ":")):
    ax3.plot(rows, rowmed[name] / np.nanmean(rowmed[name]), color=col, ls=ls, lw=0.9, label=name)
ax3.plot(rows, roi_ratio / roi_ratio.mean(), color="0.6", lw=0.8, label="ROI sums")
ax3.set_ylabel("Ga/As (row median, rel.)"); ax3.set_xlabel("map row (acquisition order)")
h3, l3 = ax3.get_legend_handles_labels()
for a in (ax1, ax2, ax3):
    for s in STEPS:
        a.axvline(s - 0.5, color=C["model"], lw=0.7, ls="--")
plt.setp(ax1.get_xticklabels(), visible=False); plt.setp(ax2.get_xticklabels(), visible=False)
ax4 = fig.add_subplot(gs[:, 1])
im = ax4.imshow(100 * wc, origin="lower", cmap="RdBu_r", vmin=-5, vmax=5, aspect="equal")
for s in STEPS:
    ax4.axhline(s - 0.5, color="k", lw=0.6, ls="--")
ax4.set_xlabel("fast (pixel)"); ax4.set_ylabel("slow (pixel)")
ax4.set_title("Residual width change (%)"); fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
panel(ax4, "b")
fig.legend(h3, l3, loc="lower left", bbox_to_anchor=(0.63, 0.03), ncol=2, fontsize=6.5,
           title="Ga/As row median (panel a, bottom)", title_fontsize=6.5)
save(fig, "fig08_resolution")

# --- fig 9: maps
XX = (scan.counters[scan.fast_motor_name] - scan.counters[scan.fast_motor_name].min()).reshape(scan.scan_shape) * 1e3
YY = (scan.counters[scan.slow_motor_name] - scan.counters[scan.slow_motor_name].min()).reshape(scan.scan_shape) * 1e3
fig, axs = plt.subplots(2, 2, figsize=(TEXTW, 5.0), gridspec_kw={"hspace": 0.38, "wspace": 0.5})
for ax, Z, ttl, cm, let in ((axs[0, 0], MB.area_map("Ga K", total=True), "Ga K area (counts / live s)", "viridis", "a"),
                            (axs[0, 1], MB.area_map("As K", total=True), "As K area (counts / live s)", "viridis", "b"),
                            (axs[1, 0], XMAP, "$n_{Ga}/n_{As}$ in the film", "cividis", "c")):
    lo, hi = np.nanpercentile(Z, [1, 99])
    im = ax.pcolormesh(XX, YY, Z, cmap=cm, shading="nearest", vmin=lo, vmax=hi)
    ax.set_aspect("equal"); ax.set_title(ttl); ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); panel(ax, let)
ax = axs[1, 1]
ax.hist(vn, bins=60, color="0.8", histtype="stepfilled", label="one resolution")
ax.hist(v, bins=60, color=C["accent"], histtype="step", lw=1.0, label="resolution tracked")
ax.axvline(np.median(v), color="k", lw=0.7, ls="--")
ax.set_xlabel("$n_{Ga}/n_{As}$"); ax.set_ylabel("pixels"); ax.legend(loc="upper left", fontsize=6)
ax.set_title("Distribution over the map"); panel(ax, "d")
save(fig, "fig09_maps")

# --- fig 10: sensitivity
labs = [k for k in R["sensitivity"] if k != "nominal"]
nom = R["sensitivity"]["nominal"]
fig, axs = plt.subplots(1, 2, figsize=(TEXTW, 3.3), sharey=True, gridspec_kw={"wspace": 0.08})
yy = np.arange(len(labs))
for ax, key, ttl in ((axs[0], "Ga", "$n_{Ga}/n_{As}$"), (axs[1], "closure", "$(n_{Ga}+n_{In})/n_{As}$")):
    val = [R["sensitivity"][k]["Ga"] + (R["sensitivity"][k]["In"] if key == "closure" else 0) for k in labs]
    ref = nom["Ga"] + (nom["In"] if key == "closure" else 0)
    ax.hlines(yy, ref, val, color="0.6", lw=1)
    ax.plot(val, yy, "o", ms=3.5, color=C["accent"])
    ax.axvline(ref, color="k", lw=0.7)
    if key == "closure":
        ax.axvline(1.0, color=C["As"], lw=0.8, ls="--")
        ax.text(1.01, len(labs) - 1, "stoichiometry", fontsize=6, color=C["As"], va="center", rotation=90)
    ax.set_title(ttl); ax.margins(x=0.08)
axs[0].set_yticks(yy); axs[0].set_yticklabels(labs); axs[0].invert_yaxis()
panel(axs[0], "a"); panel(axs[1], "b")
save(fig, "fig10_sensitivity")

R["runtime_s"] = time.time() - T0
(HERE / "results.json").write_text(json.dumps(R, indent=1, default=float, ensure_ascii=False))
print(f"done in {R['runtime_s']:.0f} s; figures in {FIG}")
