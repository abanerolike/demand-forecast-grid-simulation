"""37-bus design case of Glover, Overbye & Sarma, Chapter 10 (Problem 10.64), with sequence data.

Network, dispatch and PowerWorld results copied from the case file (pu on 100 MVA). The builder and
the Mvar-limit solver follow the Chapter 6 version of the same system; the data differ (three line
impedances, one load, the generator dispatch and CEDAR69 in service).
"""
import copy
import math

import numpy as np
import pandas as pd
import pandapower as pp

S_BASE = 100.0
TOL_MVA = 0.1
F_HZ = 60.0
SLACK_BUS = 31
SLACK_ANGLE = -14.78

BUSES = [
    (1, 'OAK345', 345.0),
    (3, 'ASH138', 138.0),
    (5, 'POPLAR69', 69.0),
    (10, 'PINE69', 69.0),
    (12, 'OAK69', 69.0),
    (13, 'PALM69', 69.0),
    (14, 'REDBUD69', 69.0),
    (15, 'PECAN69', 69.0),
    (16, 'ORANGE69', 69.0),
    (17, 'BUCKEYE69', 69.0),
    (18, 'WALNUT69', 69.0),
    (19, 'APPLE69', 69.0),
    (20, 'OLIVE69', 69.0),
    (21, 'WILLOW69', 69.0),
    (24, 'SPRUCE69', 69.0),
    (27, 'MAPLE69', 69.0),
    (28, 'ELM345', 345.0),
    (29, 'ELM138', 138.0),
    (30, 'TULIP138', 138.0),
    (31, 'SLACK345', 345.0),
    (32, 'LEMON138', 138.0),
    (33, 'LEMON69', 69.0),
    (34, 'CHERRY69', 69.0),
    (35, 'SLACK138', 138.0),
    (37, 'LIME69', 69.0),
    (38, 'PINE345', 345.0),
    (39, 'PINE138', 138.0),
    (40, 'OAK138', 138.0),
    (41, 'PEACH138', 138.0),
    (44, 'PEACH69', 69.0),
    (47, 'CEDAR138', 138.0),
    (48, 'CEDAR69', 69.0),
    (50, 'BIRCH69', 69.0),
    (53, 'PEAR138', 138.0),
    (54, 'PEAR69', 69.0),
    (55, 'LOCUST69', 69.0),
    (56, 'PLUM138', 138.0),
]

# bus, id, in service, MW, voltage setpoint [pu], Qmin, Qmax [Mvar], R, X'' [pu on 100 MVA]
GENS = [
    (14, '1', 0, 0.0, 1.02, 0.0, 5.0, 0.00163, 0.3),
    (28, '1', 1, 150.0, 1.03, -60.0, 60.0, 0.0, 0.1),
    (28, '2', 1, 150.0, 1.03, -60.0, 60.0, 0.0, 0.1),
    (31, '1', 1, 220.03927, 1.03, -90.0, 120.0, 0.0, 0.07),
    (44, '1', 1, 20.0, 1.02, -20.0, 40.0, 0.004, 0.11),
    (48, '1', 1, 16.0, 1.0, -14.0, 26.0, 0.00156, 0.25),
    (50, '1', 1, 38.0, 1.02, -18.0, 33.0, 0.0035, 0.25),
    (53, '1', 1, 140.0, 1.02, 0.0, 45.0, 0.00132, 0.12),
    (54, '1', 1, 91.48077, 1.01, -20.0, 60.0, 0.00112, 0.15),
]

# bus, constant-power MW, Mvar, constant-current MW
LOADS = [
    (3, 12.3, 5.0, 0.0),
    (5, 15.3, 3.2, 0.0),
    (10, 16.8, 2.5, 0.0),
    (12, 22.9, 6.5, 0.0),
    (13, 21.2, 6.8, 0.0),
    (14, 22.2, 15.2, 0.0),
    (15, 58.2, 36.3, 0.0),
    (16, 57.8, 40.4, 0.0),
    (17, 32.8, 12.9, 0.0),
    (18, 51.2, 15.2, 0.0),
    (19, 18.3, 5.0, 0.0),
    (20, 15.3, 5.0, 0.0),
    (21, 37.2, 13.4, 0.0),
    (24, 36.3, 10.4, 0.0),
    (27, 19.5, 7.8, 0.0),
    (30, 23.4, 6.2, 0.0),
    (33, 14.0, 3.0, 0.0),
    (34, 34.9, 0.0, 9.7),
    (37, 33.4, 9.6, 0.0),
    (44, 59.8, 12.3, 0.0),
    (48, 55.8, 12.5, 0.0),
    (50, 14.1, 2.0, 0.0),
    (53, 59.5, 27.8, 0.0),
    (54, 12.43, 5.73, 0.0),
    (55, 45.3, 12.3, 0.0),
    (56, 14.0, 3.7, 0.0),
]

# from, to, circuit, R, X, B [pu], limit [MVA]
LINES = [
    (1, 31, '1', 0.00117, 0.017, 0.182, 597.0),
    (3, 40, '1', 0.0048, 0.0368, 0.0182, 233.0),
    (3, 41, '1', 0.00902, 0.0571, 0.0135, 233.0),
    (5, 18, '1', 0.03133, 0.07675, 0.0015, 82.0),
    (5, 44, '1', 0.03463, 0.08253, 0.0016, 82.0),
    (10, 13, '1', 0.03375, 0.0789, 0.0012, 82.0),
    (10, 19, '1', 0.0405, 0.0953, 0.0021, 72.0),
    (12, 17, '1', 0.02172, 0.06935, 0.0019, 82.0),
    (12, 18, '1', 0.02747, 0.08909, 0.0019, 106.0),
    (12, 27, '1', 0.0142, 0.07557, 0.0695, 112.0),
    (13, 55, '1', 0.02216, 0.0904, 0.0017, 72.0),
    (14, 34, '1', 0.02625, 0.06429, 0.0012, 72.0),
    (14, 44, '1', 0.04211, 0.08545, 0.0474, 81.0),
    (15, 16, '1', 0.00753, 0.02582, 0.0206, 91.0),
    (15, 24, '1', 0.01835, 0.02845, 0.0221, 74.0),
    (15, 54, '1', 0.00846, 0.00465, 0.021, 68.0),
    (15, 54, '2', 0.00855, 0.0047, 0.0213, 68.0),
    (15, 54, '3', 0.00859, 0.00472, 0.0214, 68.0),
    (16, 27, '1', 0.00366, 0.01312, 0.046, 112.0),
    (17, 19, '1', 0.02703, 0.03613, 0.002, 50.0),
    (18, 37, '1', 0.01017, 0.00559, 0.0253, 68.0),
    (18, 37, '2', 0.01017, 0.00559, 0.0253, 68.0),
    (20, 34, '1', 0.02423, 0.05862, 0.0011, 72.0),
    (20, 48, '1', 0.02133, 0.0521, 0.0009, 82.0),
    (20, 50, '1', 0.0424, 0.07509, 0.0008, 82.0),
    (21, 48, '1', 0.01824, 0.04236, 0.0075, 72.0),
    (21, 48, '2', 0.01829, 0.04246, 0.0078, 72.0),
    (24, 44, '1', 0.03993, 0.09965, 0.002, 82.0),
    (31, 28, '1', 0.00224, 0.03268, 0.35, 600.0),
    (32, 29, '1', 0.0103, 0.05681, 0.0164, 191.0),
    (29, 41, '1', 0.01868, 0.1259, 0.0356, 191.0),
    (56, 29, '1', 0.00771, 0.0544, 0.0138, 233.0),
    (30, 32, '1', 0.00225, 0.0134, 0.004, 191.0),
    (30, 41, '1', 0.00735, 0.04395, 0.0123, 191.0),
    (31, 38, '1', 0.00075, 0.01092, 0.117, 597.0),
    (33, 50, '1', 0.0502, 0.101, 0.0025, 82.0),
    (35, 39, '1', 0.01028, 0.07253, 0.0176, 288.0),
    (35, 56, '1', 0.01243, 0.07847, 0.0192, 233.0),
    (39, 40, '1', 0.01161, 0.08085, 0.0225, 233.0),
    (39, 47, '1', 0.00775, 0.05244, 0.0154, 233.0),
    (47, 53, '1', 0.00204, 0.00692, 0.1601, 185.0),
    (48, 54, '1', 0.01473, 0.036, 0.0744, 83.0),
    (54, 55, '1', 0.06246, 0.08246, 0.0003, 50.0),
]

# from, to, circuit, R, X, B [pu], off-nominal tap on the from side, limit [MVA]
TRAFOS = [
    (1, 40, '1', 0.001, 0.0623, 0.0, 1.0, 250.0),
    (10, 39, '1', 0.00106, 0.03949, -0.0041, 1.0125, 186.7),
    (12, 40, '1', 0.00107, 0.04039, -0.0042, 1.025, 187.0),
    (12, 40, '2', 0.00107, 0.04039, -0.0042, 1.025, 187.0),
    (28, 29, '1', 0.00087, 0.051, 0.0, 1.0, 220.0),
    (28, 29, '2', 0.00087, 0.051, 0.0, 1.0, 220.0),
    (35, 31, '1', 0.00087, 0.051, 0.0, 1.0, 220.0),
    (33, 32, '1', 0.0025, 0.0723, 0.0, 1.0, 101.0),
    (39, 38, '1', 0.00094, 0.05107, -0.0127, 1.0, 224.0),
    (39, 38, '2', 0.00095, 0.05116, -0.0124, 1.0, 224.0),
    (44, 41, '1', 0.0025, 0.07144, -0.0045, 1.02125, 101.0),
    (44, 41, '2', 0.00244, 0.06829, -0.006, 1.02125, 101.0),
    (48, 47, '1', 0.00101, 0.03925, -0.0035, 1.05, 187.0),
    (54, 53, '1', 0.00134, 0.04988, -0.0004, 1.0, 187.0),
]

# bus, Mvar of the switched shunt in its saved position
SHUNTS = [
    (13, 4.8),
    (14, 7.2),
    (15, 12.6),
    (16, 28.8),
    (17, 15.6),
    (18, 14.4),
    (20, 7.2),
    (44, 0.0),
]

# transformer core loss modelled as a line shunt: from, to, circuit, location bus, MW
LINE_SHUNTS = [
    (12, 40, '1', 40, 0.06),
    (12, 40, '2', 40, 0.06),
    (39, 38, '1', 38, 0.14),
    (39, 38, '2', 38, 0.14),
    (44, 41, '1', 41, 0.07),
    (44, 41, '2', 41, 0.07),
    (48, 47, '1', 47, 0.06),
    (54, 53, '1', 53, 0.04),
]

# PowerWorld solved state saved in the case, bus order as in BUSES
PW_VM = [1.02396722, 1.00175334, 1.00406256, 1.02186552, 1.02045487, 1.00032572, 1.00290421, 1.00625819, 1.00371111, 1.01230544, 0.99706328, 1.01261962, 1.0105586, 1.01189657, 1.00270509, 1.00567627, 1.02999998, 1.02448662, 1.00825743, 1.02999997, 1.01129384, 1.01634406, 1.00117735, 1.02482714, 0.9951572, 1.02468083, 1.01429131, 1.00298947, 1.00270026, 1.0200001, 0.99759722, 1.01791959, 1.02000013, 1.00089621, 1.01000042, 0.99017701, 1.02337534]
PW_VA = [-15.8460033, -20.1724398, -24.0086338, -20.1442202, -21.5788746, -21.9746086, -23.7910601, -22.4275052, -23.0562326, -22.2773614, -24.8165321, -21.8832611, -22.8123043, -22.8729781, -22.8663323, -22.9618797, -12.8041092, -15.4089066, -19.0582701, -14.78, -18.5431042, -19.7570376, -24.0555577, -16.0902864, -24.8496067, -15.7499015, -18.0764214, -19.9055093, -20.2021799, -22.5126343, -20.3877233, -22.4982808, -20.9546117, -20.3911429, -22.3370906, -23.1259761, -15.9280896]

def build_case():
    """Positive-sequence network in its saved state (controls frozen)."""
    net = pp.create_empty_network(sn_mva=S_BASE, f_hz=F_HZ)
    kv = {}
    for num, name, nom in BUSES:
        pp.create_bus(net, vn_kv=nom, name=name, index=num)
        kv[num] = nom
    for bus, gid, on, p, vset, qmin, qmax, r, x in GENS:
        if bus == SLACK_BUS:
            pp.create_ext_grid(net, bus, vm_pu=vset, va_degree=SLACK_ANGLE, name=f"G{bus}-{gid}")
            continue
        pp.create_gen(net, bus, p_mw=p, vm_pu=vset, min_q_mvar=qmin, max_q_mvar=qmax,
                      in_service=bool(on), name=f"G{bus}-{gid}")
    for bus, p, q, p_i in LOADS:
        pp.create_load(net, bus, p_mw=p + p_i, q_mvar=q, const_i_p_percent=100 * p_i / (p + p_i),
                       name=f"L{bus}")
    for f, t, ckt, r, x, b, lim in LINES:
        z = kv[f] ** 2 / S_BASE
        pp.create_line_from_parameters(
            net, f, t, length_km=1.0, r_ohm_per_km=r * z, x_ohm_per_km=x * z,
            c_nf_per_km=b / z / (2 * math.pi * F_HZ) * 1e9, max_i_ka=lim / (math.sqrt(3) * kv[f]),
            name=f"{f}-{t}-{ckt}")
    for f, t, ckt, r, x, b, tap, lim in TRAFOS:
        hv, lv = (f, t) if kv[f] > kv[t] else (t, f)
        rated = {hv: kv[hv], lv: kv[lv]}
        rated[f] = kv[f] * tap
        # magnetizing B is negative in the case file; i0_percent with the pi model reproduces it
        pp.create_transformer_from_parameters(
            net, hv, lv, sn_mva=S_BASE, vn_hv_kv=rated[hv], vn_lv_kv=rated[lv],
            vkr_percent=r * 100, vk_percent=math.hypot(r, x) * 100, pfe_kw=0.0,
            i0_percent=abs(b) * 100, shift_degree=0.0, name=f"{f}-{t}-{ckt}")
    for bus, mvar in SHUNTS:
        pp.create_shunt(net, bus, q_mvar=-mvar, p_mw=0.0, name=f"SS{bus}")
    for f, t, ckt, loc, mw in LINE_SHUNTS:
        pp.create_shunt(net, loc, q_mvar=0.0, p_mw=mw, name=f"LS{f}-{t}-{ckt}")
    net.gen["pv_on"] = net.gen.in_service
    return net


def solve_pw(net, vtol=5e-5, max_outer=50, **kw):
    """Power flow with PowerWorld's Mvar-limit back-off: a unit parked on a limit becomes a fixed-Q
    injection and returns to voltage control once its bus voltage crosses the setpoint on the side
    the limit cannot explain."""
    pv = net.gen.index[net.gen.pv_on.astype(bool)].tolist()
    parked = {}
    for _ in range(max_outer):
        net.sgen.drop(net.sgen.index, inplace=True)
        net.gen.loc[pv, "in_service"] = True
        for i, side in parked.items():
            net.gen.at[i, "in_service"] = False
            pp.create_sgen(net, net.gen.at[i, "bus"], p_mw=net.gen.at[i, "p_mw"],
                           q_mvar=net.gen.at[i, f"{side}_q_mvar"], name=f"park{i}")
        pp.runpp(net, enforce_q_lims=False, trafo_model="pi", **kw)
        vb = net.res_bus.vm_pu
        switch = None
        for i in pv:
            vset, b = net.gen.at[i, "vm_pu"], net.gen.at[i, "bus"]
            if i in parked:
                if (parked[i] == "max" and vb[b] > vset + vtol) or (parked[i] == "min" and vb[b] < vset - vtol):
                    switch = (i, None)
            elif net.res_gen.at[i, "q_mvar"] > net.gen.at[i, "max_q_mvar"] + 1e-6:
                switch = (i, "max")
            elif net.res_gen.at[i, "q_mvar"] < net.gen.at[i, "min_q_mvar"] - 1e-6:
                switch = (i, "min")
            if switch:
                break
        if switch is None:
            net["parked"] = dict(parked)
            return net
        i, side = switch
        if side is None:
            del parked[i]
        else:
            parked[i] = side
    raise RuntimeError("Mvar limit switching did not settle")


def residual_band(build, measure, tol=TOL_MVA, **kw):
    """Worst-case shift of every measured quantity when any non-slack bus keeps a mismatch up to
    the MVA tolerance (the accuracy of PowerWorld's own solution)."""
    net = build()
    base = measure(solve_pw(net, **kw))
    band = pd.Series(0.0, index=base.index)
    pv_buses = set(net.gen.bus[net.gen.pv_on.astype(bool)])
    for bus in net.bus.index.difference([SLACK_BUS]):
        for col in (["p_mw"] if bus in pv_buses else ["p_mw", "q_mvar"]):
            worst = pd.Series(0.0, index=base.index)
            for sign in (1, -1):
                extra = pp.create_load(net, bus, **{"p_mw": 0.0, "q_mvar": 0.0, col: sign * tol})
                worst = np.maximum(worst, (measure(solve_pw(net, **kw)) - base).abs())
                net.load.drop(extra, inplace=True)
            band += worst
    return band


def compare(rows):
    """Comparison table; fails loudly when any row is outside its tolerance."""
    df = pd.DataFrame(rows, columns=["Group", "Quantity", "pandapower", "Reference", "Tolerance", "Source"])
    df["Abs diff"] = (df.pandapower - df.Reference).abs()
    df["Pass"] = df["Abs diff"] <= df.Tolerance
    assert df.Pass.all(), df.loc[~df.Pass, "Quantity"].tolist()
    return df


def summarize(df):
    g = df.groupby(["Group", "Source"], sort=False)
    return pd.DataFrame({
        "Rows": g.size(),
        "Max abs diff": g["Abs diff"].max(),
        "Tightest tolerance": g.Tolerance.min(),
        "Smallest margin": (df.Tolerance - df["Abs diff"]).groupby([df.Group, df.Source], sort=False).min(),
        "All pass": g.Pass.all(),
    }).reset_index()
