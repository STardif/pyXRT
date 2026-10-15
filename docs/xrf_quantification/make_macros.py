"""
Turn ``results.json`` (written by ``make_figures.py``) into ``results_macros.tex``.

Every number quoted in ``xrf_quantification.tex`` is a macro defined here, so
the text cannot drift from the computation: rerun ``make_figures.py`` then
this script, and recompile.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
out = []


def m(name, value):
    assert name.isalpha(), name
    out.append(f"\\newcommand{{\\r{name}}}{{{value}}}")


def f(x, n=3):
    return f"{x:.{n}f}"


def pct(x, n=1):
    return f"{100 * x:.{n}f}"


def num(x, sig=3):
    """Number in siunitx form, e.g. 4.80e8 -> \\num{4.80e8}."""
    return f"\\num{{{x:.{sig - 1}e}}}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")


D, B, DET, PU, Q, FIT = R["data"], R["beam"], R["detector"], R["pileup"], R["quant"], R["fit"]
RES, MAPS = R["resolution"], R["maps"]

# --- data
m("NPix", D["n_pixels"]); m("NSlow", D["shape"][0]); m("NFast", D["shape"][1])
m("StepUm", f(D["step_um"], 2)); m("CountTime", f(D["count_time_s"], 0))
m("CountsPerPixel", num(D["counts_per_pixel"])); m("TotalCounts", num(D["total_counts"]))
m("IcrMin", f(D["icr_kcps"][0], 1)); m("IcrMean", f(D["icr_kcps"][1], 1)); m("IcrMax", f(D["icr_kcps"][2], 1))
m("OcrMean", f(D["ocr_kcps_mean"], 1))
m("DeadMin", f(D["dead_time"][0], 3)); m("DeadMed", f(D["dead_time"][1], 3)); m("DeadMax", f(D["dead_time"][2], 3))
m("LiveMin", f(D["live_time_s"][0], 3)); m("LiveMax", f(D["live_time_s"][1], 3))
m("RealTime", f(D["real_time_s_mean"], 3)); m("MonSpread", f(D["monitor_spread_pct"], 1))
m("ScanHours", f(D["scan_duration_s"] / 3600, 2)); m("ScanSec", f(D["scan_duration_s"], 0))
m("RoiLo", D["roi_channels"][0]); m("RoiHi", D["roi_channels"][1])
m("CalGain", f(D["calibration"][0], 3)); m("CalOffset", f(D["calibration"][1], 3))
m("RateMonCorr", f(D["rate_monitor_corr"], 3)); m("RateRowSpread", f(D["rate_row_spread_pct"], 2))
m("RateMonRowSpread", f(D["rate_over_monitor_row_spread_pct"], 2)); m("GaRowSpread", f(D["ga_area_row_spread_pct"], 2))
for i, rf in enumerate(R["refills"]):
    L = "ABC"[i]
    m(f"Refill{L}Row", rf["row"]); m(f"Refill{L}Col", rf["col"]); m(f"Refill{L}Time", f(rf["time_s"], 0))
    m(f"Refill{L}Mon", f(rf["monitor_jump_pct"], 1)); m(f"Refill{L}Rate", f(rf["raw_rate_jump_pct"], 1))
    m(f"Refill{L}Ga", f(rf["ga_area_step_pct"], 2)); m(f"Refill{L}As", f(rf["as_area_step_pct"], 2))
    m(f"Refill{L}Ratio", f(rf["ratio_step_pct"], 2))

# --- beam
m("Ec", f(B["ec_kev"], 2)); m("BandTenLo", f(B["band10_kev"][0], 1)); m("BandTenHi", f(B["band10_kev"][1], 1))
m("BandOneLo", f(B["band1_kev"][0], 1)); m("BandOneHi", f(B["band1_kev"][1], 1))
m("PeakBW", f(B["peak_bw_kev"], 1)); m("PeakKeV", f(B["peak_kev_kev"], 1))
m("kBW", f(B["k_gaas_per_bw"], 3)); m("kKeV", f(B["k_gaas_per_kev"], 3))
m("kJacobianBias", f(100 * (B["k_gaas_per_bw"] / B["k_gaas_per_kev"] - 1), 1))
m("LorentzGamma", f(B["gamma"], 0)); m("InvGamma", f(B["inv_gamma_urad"], 1))
for e0, L_ in (("5", "Five"), ("10", "Ten"), ("20", "Twenty"), ("30", "Thirty")):
    m(f"FwhmPsi{L_}", f(B["fwhm_psi_urad"][e0], 0))
m("CurrentMA", f(B["current_ma"], 1)); m("FillMode", B["filling_mode"])
st = B["settings"]
for k_, L_ in (("vg1", "VgOne"), ("hg1", "HgOne"), ("vg2", "VgTwo"), ("hg2", "HgTwo"), ("vg3", "VgThree"), ("hg3", "HgThree")):
    m(f"Set{L_}", f(st[k_], 2))
m("SetMaOne", f(st["ma1"], 3)); m("SetMaTwo", f(st["ma2"], 3))
m("SetRmOne", f(st["mc1_km"], 1)); m("SetRmTwo", f(st["mc2_km"], 1))
vl = {k.split(" ")[0]: v for k, v in B["v_limits"].items()}
hl = {k.split(" ")[0]: v for k, v in B["h_limits"].items()}
for k_, L_ in (("s1", "SOne"), ("m1", "MOne"), ("m2", "MTwo"), ("s2", "STwo"), ("s3", "SThree"), ("KB-V", "KBV")):
    m(f"VAcc{L_}", f(vl[k_], 1))
for k_, L_ in (("s1", "SOne"), ("s2", "STwo"), ("s3", "SThree"), ("KB-H", "KBH")):
    m(f"HAcc{L_}", f(hl[k_], 1))
m("VAcc", f(B["v_acc_urad"], 2)); m("HAcc", f(B["h_acc_urad"], 2))
m("VLim", B["v_limited_by"].split(" ")[0]); m("HLim", B["h_limited_by"].split(" ")[0])
m("FocMOne", f(B["focal_m1_m"], 0)); m("FocMTwo", f(B["focal_m2_m"], 0)); m("DivGain", f(B["divergence_gain"], 3))
m("HeightKBV", f(B["height_per_urad_kbv"], 1))
m("KBSpreadV", f(B["kb_spread_v_mrad"], 2)); m("KBSpreadH", f(B["kb_spread_h_mrad"], 2))
m("KBBeamV", f(B["kb_beam_v_mm"], 2)); m("KBBeamH", f(B["kb_beam_h_mm"], 2))
m("KBFootV", f(B["kb_footprint_v_mm"], 0)); m("KBFootH", f(B["kb_footprint_h_mm"], 0))
m("PeakKeVFull", f(B["peak_kev_full"], 1)); m("MeanKeV", f(B["mean_kev"], 2)); m("MeanKeVFull", f(B["mean_kev_full"], 2))
m("BandTenFullLo", f(B["band10_full_kev"][0], 1)); m("BandTenFullHi", f(B["band10_full_kev"][1], 1))
m("kFull", f(B["k_gaas_full"], 3)); m("kSourceShift", f(100 * (B["k_gaas_per_kev"] / B["k_gaas_full"] - 1), 1))
for o_, L_ in (("10.0", "Ten"), ("25.0", "TwentyFive"), ("50.0", "Fifty")):
    m(f"Off{L_}", f(100 * (B["off_scan"][o_] / B["k_gaas_per_kev"] - 1), 2))
m("FluxTwenty", num(B["flux_20kev_ph_s_bw"], 2))
m("KBSpreadEffect", f(100 * (B["k_gaas_per_kev"] / B["k_gaas_no_kb_spread"] - 1), 2))
m("AirForClosure", f(R["air_for_closure_cm"], 1))
m("AccRatioLo", f(B["fwhm_psi_urad"]["30"] / B["v_acc_urad"], 0)); m("AccRatioHi", f(B["fwhm_psi_urad"]["5"] / B["v_acc_urad"], 0))

# --- detector
m("EffIn", f(DET["In La1"], 4)); m("EffGa", f(DET["Ga Ka1"], 3)); m("EffAs", f(DET["As Ka1"], 3))
m("EffAsKb", f(DET["As Kb1"], 3)); m("EffInPerCm", f(DET["In_La1_per_cm"], 3))
m("EffGaPerCm", f(DET["Ga_Ka1_per_cm"], 3)); m("EffInLowP", f(DET["In_La1_990hPa"], 3))
m("EscGa", pct(DET["escape_Ga_Ka1"], 3)); m("EscIn", pct(DET["escape_In_La1"], 2))
m("EscMax", pct(DET["escape_max"], 2)); m("EscMaxE", f(DET["escape_max_ev"] / 1000, 2))

# --- pile-up
m("Tau", f(PU["tau_us"], 3)); m("TauNs", f(1000 * PU["tau_us"], 0)); m("TauErr", f(PU["tau_err_us"], 4))
m("TauChiWith", f(PU["chi2_with"], 0)); m("TauChiWithout", f(PU["chi2_without"], 0))
m("PairProb", pct(PU["pair_probability"], 1))

# --- physics levels
for key, L in (("scalar", "Scalar"), ("white", "White"), ("white+eff", "WhiteEff"), ("matrix", "Matrix")):
    m(f"Chi{L}", f(R["levels"][key]["chi2"], 0))
    il = R["In_L_rel"][key]
    m(f"InLb{L}", f(il["L2M4"], 3)); m(f"InLbiii{L}", f(il["L3N5"], 3)); m(f"InLbiv{L}", f(il["L1M3"], 3))
    m(f"AsKb{L}", f(R["As_K_rel"][key]["KM3"], 4)); m(f"AsKaii{L}", f(R["As_K_rel"][key]["KL2"], 4))
E = R["escape_test"]
m("NiNoEsc", f(E["Ni_without"], 1)); m("NiEsc", f(E["Ni_with"], 1))
m("ChiNoEsc", f(E["chi2_without"], 0)); m("ChiEsc", f(E["chi2_with"], 0))

# --- fit
m("Chi", f(FIT["chi2"], 0)); m("NChan", FIT["n_channels"]); m("NFree", FIT["n_free"])
m("NLines", FIT["n_lines"]); m("NGroups", FIT["n_groups"])
m("Noise", f(FIT["noise_ev"], 1)); m("Fano", f(FIT["fano"], 3)); m("FanoW", f(FIT["fano"] * 3.85 / 3.66, 3))
m("Gamma", f(FIT["gamma_ev"], 1)); m("FwhmMn", f(FIT["fwhm_mnka_ev"], 0)); m("FwhmGa", f(FIT["fwhm_gaka_ev"], 0))
m("ResidRms", f(FIT["resid_rms_pct"], 2))
for lab, L in (("In L", "In"), ("Ga K", "Ga"), ("As K", "As"), ("Cu K", "Cu"), ("Ni K", "Ni"),
               ("Ca K", "Ca"), ("Ar K", "Ar")):
    a, e = FIT["areas"][lab], FIT["area_errors"][lab]
    m(f"Area{L}", f(a, 2) if a < 1000 else f(a, 0)); m(f"AreaErr{L}", f(e, 2) if e < 1 else f(e, 1))
    m(f"Snr{L}", f(MAPS["snr"][lab], 1) if MAPS["snr"][lab] < 20 else f(MAPS["snr"][lab], 0))

# --- quantification
m("xGa", f(Q["ratios"]["Ga"], 4)); m("xIn", f(Q["ratios"]["In"], 3)); m("Closure", f(Q["closure"], 3))
m("Iter", Q["iterations"]); m("kGa", f(Q["k"]["Ga K"], 4)); m("kIn", f(Q["k"]["In L"], 1))
m("SFGa", pct(Q["enhancement"]["Ga K"], 2)); m("SFIn", pct(Q["enhancement"]["In L"], 2))
m("SFAs", pct(Q["enhancement"]["As K"], 3))
m("AbsGa", f(Q["absorption"]["Ga K"], 3)); m("AbsAs", f(Q["absorption"]["As K"], 3)); m("AbsIn", f(Q["absorption"]["In L"], 3))
m("CapInFrac", pct(Q["other_layers"]["In L"], 1)); m("CapInSF", pct(Q["cap_In_SF_over_primary"], 1))
m("CapTIn", f(Q["cap_transmission"]["In La1"], 3)); m("CapTGa", f(Q["cap_transmission"]["Ga Ka1"], 3))
m("CapTAs", f(Q["cap_transmission"]["As Ka1"], 3))
sg = Q["Ga_SF_by_exciter"]
m("SFGaAsKLiii", pct(sg["1:As KL3"], 2)); m("SFGaAsKLii", pct(sg["1:As KL2"], 2))
m("SFGaAsKMiii", pct(sg["1:As KM3"], 2))
si = Q["In_SF_by_exciter"]
m("SFInAsKLiii", pct(si["1:As KL3"], 2)); m("SFInGaKLiii", pct(si["1:Ga KL3"], 2))
m("xGaNoCap", f(Q["no_cap"]["Ga"], 4)); m("xInNoCap", f(Q["no_cap"]["In"], 3))
m("ClosureNoCap", f(Q["no_cap"]["Ga"] + Q["no_cap"]["In"], 3))
m("xGaNoMatrix", f(Q["no_matrix"]["Ga"], 4)); m("xInNoMatrix", f(Q["no_matrix"]["In"], 3))
m("ClosureNoMatrix", f(Q["no_matrix"]["Ga"] + Q["no_matrix"]["In"], 3))
cs = R["cap_scan"]
for tn, L in (("0", "Zero"), ("100", "One"), ("200", "Two"), ("300", "Three"), ("400", "Four"), ("500", "Five")):
    m(f"CapScan{L}", f(cs[tn]["Ga"] + cs[tn]["In"], 3))

# sensitivity table body
rows = []
nom = R["sensitivity"]["nominal"]
for lab, v in R["sensitivity"].items():
    lab_tex = (lab.replace("µm", "\\si{\\micro\\metre}").replace("°", "\\si{\\degree}")
               .replace("µrad", "\\si{\\micro\\radian}").replace("ψ", "$\\psi$").replace("G1", "$G_1$"))
    rows.append(f"{lab_tex} & {v['Ga']:.4f} & \\num{{{(round(100 * (v['Ga'] / nom['Ga'] - 1), 1) + 0.0):+.1f}}} & {v['In']:.3f} & "
                f"{v['Ga'] + v['In']:.3f} \\\\")
out.append("\\newcommand{\\rSensitivityRows}{%\n" + "\n".join(rows) + "}")
PLAUS = {k: v for k, v in R["sensitivity"].items() if not k.startswith("source G1")}
vals = [v["Ga"] for v in PLAUS.values()]
m("xGaMin", f(min(vals), 3)); m("xGaMax", f(max(vals), 3))
cl = [v["Ga"] + v["In"] for v in PLAUS.values()]
m("ClosureMin", f(min(cl), 2)); m("ClosureMax", f(max(cl), 2))
S = R["sensitivity"]
G1c = R["sensitivity"]["source G1 (all ψ)"]
m("xGaGOne", f(G1c["Ga"], 4)); m("xInGOne", f(G1c["In"], 3)); m("ClosureGOne", f(G1c["Ga"] + G1c["In"], 3))
m("GOneShift", f(100 * (R["quant"]["ratios"]["Ga"] / G1c["Ga"] - 1), 1))
for k_, L_ in (("KB angles 2.5 mrad", "KBLo"), ("KB angles 3.5 mrad", "KBHi"), ("beam offset 25 µrad", "OffTF")):
    m(f"xGa{L_}", f(R["sensitivity"][k_]["Ga"], 4)); m(f"Closure{L_}", f(R["sensitivity"][k_]["Ga"] + R["sensitivity"][k_]["In"], 3))
m("ClosureAirLo", f(S["air 18 cm"]["Ga"] + S["air 18 cm"]["In"], 2))
m("ClosureAirHi", f(S["air 22 cm"]["Ga"] + S["air 22 cm"]["In"], 2))

# --- resolution
m("WinLo", f(RES["window_ev"][0], 0)); m("WinHi", f(RES["window_ev"][1], 0))
m("StepRow", RES["steps"][0]); m("NCentroidSteps", len(RES["centroid_steps"]))
m("WidthErr", f(RES["width_err_ev"], 2)); m("WidthBefore", f(RES["width_before"], 1))
m("WidthAfter", f(RES["width_after"], 1))
m("WidthJump", f(100 * (RES["width_after"] / RES["width_before"] - 1), 1))
m("CentroidRange", f(RES["centroid_range_ev"], 1))
m("FwhmSegA", f(RES["fwhm_segments"]["0"], 1)); m("FwhmSegB", f(RES["fwhm_segments"]["1"], 1))
m("GammaSegA", f(RES["gamma_segments"]["0"], 1)); m("GammaSegB", f(RES["gamma_segments"]["1"], 1))
m("ChiSegA", f(RES["chi2_segments"]["0"], 0)); m("ChiSegB", f(RES["chi2_segments"]["1"], 0))
m("Invariance", num(RES["invariance"], 2)); m("ChangePix", RES["change_pixel_in_row"])
m("ChangePixPrev", RES["change_pixel_in_row"] - 1); m("ChangeTime", f(RES["change_time_s"], 0))
m("WcFirst", f(RES["width_change_first_pixels_pct"][0], 1))
m("WcBeforeChange", f(RES["width_change_first_pixels_pct"][RES["change_pixel_in_row"] - 1], 1))
m("WcAtChange", f(RES["width_change_first_pixels_pct"][RES["change_pixel_in_row"]], 1))
m("RoiRatioBefore", f(RES["roi_sum_ratio_before_after"][0], 4)); m("RoiRatioAfter", f(RES["roi_sum_ratio_before_after"][1], 4))
m("IcrBefore", f(RES["icr_before_after_kcps"][0], 1)); m("IcrAfter", f(RES["icr_before_after_kcps"][1], 1))
mrows = []
for name, v in RES["modes"].items():
    mrows.append(f"\\texttt{{{name}}} & \\num{{{v['step_pct']:+.2f}}} & {v['spread_pct']:.2f} & {v['chi2_median']:.3f} & "
                 f"{v['chi2_p5_p95'][0]:.2f}--{v['chi2_p5_p95'][1]:.2f} \\\\")
out.append("\\newcommand{\\rModeRows}{%\n" + "\n".join(mrows) + "}")
for name, L in (("none", "None"), ("segments", "Seg"), ("linear", "Lin"), ("both", "Both")):
    v = RES["modes"][name]
    m(f"Step{L}", f(v["step_pct"], 2)); m(f"Spread{L}", f(v["spread_pct"], 2)); m(f"ChiMap{L}", f(v["chi2_median"], 3))

# --- maps
m("xMed", f(MAPS["x_median"], 4)); m("xSpread", f(MAPS["x_spread"], 4)); m("xStat", f(MAPS["x_stat"], 4))
m("xExcess", f(MAPS["x_excess"], 4)); m("xNoneMed", f(MAPS["x_none_median"], 4))
m("xNoneSpread", f(MAPS["x_none_spread"], 4)); m("xExcessNone", f(MAPS["x_excess_none"], 4))

# --- matrix thickness curves (selected)
mc = R["matrix_curves"]
i19 = mc["thickness_um"].index(1.9); i10 = mc["thickness_um"].index(10.0); i02 = mc["thickness_um"].index(0.2)
m("CurveAbsGaNineteen", f(mc["absorption"]["Ga Ka1"][i19], 3)); m("CurveAbsInNineteen", f(mc["absorption"]["In La1"][i19], 3))
m("CurveSFGaNineteen", pct(mc["enhancement"]["Ga Ka1"][i19], 1)); m("CurveSFInNineteen", pct(mc["enhancement"]["In La1"][i19], 1))
m("CurveSFGaTen", pct(mc["enhancement"]["Ga Ka1"][i10], 1)); m("CurveSFInTen", pct(mc["enhancement"]["In La1"][i10], 1))
m("CurveSFGaTwoHundred", pct(mc["enhancement"]["Ga Ka1"][i02], 2))
m("CurveAbsInTen", f(mc["absorption"]["In La1"][i10], 3))

# --- derived quantities quoted in the text
m("MatrixEffect", f(100 * (Q["ratios"]["Ga"] / Q["no_matrix"]["Ga"] - 1), 1))
over = [100 * (rf["monitor_jump_pct"] - rf["raw_rate_jump_pct"]) / rf["monitor_jump_pct"] for rf in R["refills"]]
m("MonOverMin", f(min(over), 0)); m("MonOverMax", f(max(over), 0))
m("InPerCmPct", f(100 * (1 - DET["In_La1_per_cm"]), 0)); m("LowPPct", f(100 * (DET["In_La1_990hPa"] - 1), 0))
m("CntErrPeak", num(D["cnt_err_over_peak"], 2)); m("SemPeak", num(D["sem_over_peak"], 2))
m("SFGaPct", pct(Q["enhancement"]["Ga K"], 1))

(HERE / "results_macros.tex").write_text("% generated by make_macros.py from results.json -- do not edit\n"
                                         + "\n".join(out) + "\n")
print(f"{len(out)} macros written")
