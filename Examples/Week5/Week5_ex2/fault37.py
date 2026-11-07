"""Three-phase fault data and helpers for the 37-bus case (Problem 8.28).

The network itself comes from case37.py; this module adds the generator
subtransient impedances and the PowerWorld fault results.
"""
import numpy as np
import pandas as pd
import pandapower.shortcircuit as sc
import pandapower.toolbox as tb

from case37 import BUSES, S_BASE, SLACK_BUS, build_case

# Positive-sequence generator impedance R + jX'' [pu on 100 MVA], by bus (both ELM345 units identical).
#    PEACH69 shows blank sequence fields; 0.004 + j0.127 is back-calculated and reproduces every
#    Thevenin impedance in the PowerWorld fault table
GEN_Z = {28: (0.0, 0.05), 31: (0.0, 0.05), 44: (0.004, 0.127), 50: (0.0035, 0.15),
         53: (0.00132, 0.05), 54: (0.00112, 0.088)}

# PowerWorld Fault Definitions, bolted 3PB fault at each bus:
#    bus, |I_F| [pu], angle [deg], Thevenin R [pu], Thevenin X [pu]
PW_FAULT = [
    (1, 33.30336, -100.84793, 0.00267, 0.03062),
    (3, 20.76826, -101.28017, 0.00740, 0.04766),
    (5, 11.83993, -93.57965, 0.02951, 0.07951),
    (10, 19.46285, -100.10355, 0.00914, 0.05170),
    (12, 25.10717, -99.57999, 0.00836, 0.03977),
    (13, 11.13177, -91.42577, 0.03153, 0.08415),
    (14, 12.18353, -91.45370, 0.03130, 0.07616),
    (15, 37.60104, -97.99065, 0.00668, 0.02591),
    (16, 31.14773, -91.28963, 0.01171, 0.02999),
    (17, 12.66894, -93.67999, 0.02536, 0.07577),
    (18, 11.59053, -94.02414, 0.03041, 0.08048),
    (19, 11.73218, -91.68787, 0.02970, 0.08103),
    (20, 18.81822, -93.68668, 0.01766, 0.05075),
    (21, 17.62520, -94.87049, 0.01798, 0.05463),
    (24, 21.56082, -90.51328, 0.01769, 0.04301),
    (27, 25.25720, -93.93317, 0.01272, 0.03770),
    (28, 64.02462, -100.63818, 0.00060, 0.01608),
    (29, 35.70180, -100.96893, 0.00221, 0.02861),
    (30, 21.10128, -99.46037, 0.00793, 0.04712),
    (31, 54.34645, -101.00743, 0.00125, 0.01891),
    (32, 22.01184, -98.99797, 0.00758, 0.04531),
    (33, 12.93089, -98.32873, 0.01546, 0.07705),
    (34, 12.40574, -91.65740, 0.03082, 0.07463),
    (35, 25.60461, -101.01644, 0.00354, 0.03987),
    (37, 11.03535, -91.60228, 0.03546, 0.08292),
    (38, 40.99099, -101.15263, 0.00200, 0.02492),
    (39, 37.87749, -100.78028, 0.00340, 0.02656),
    (40, 29.43024, -100.75124, 0.00537, 0.03365),
    (41, 27.40045, -101.28587, 0.00564, 0.03615),
    (44, 27.76019, -100.39852, 0.00769, 0.03593),
    (47, 40.39803, -102.15981, 0.00358, 0.02445),
    (48, 29.01313, -98.84859, 0.00842, 0.03412),
    (50, 16.77445, -97.31537, 0.01420, 0.05913),
    (53, 43.19292, -103.49786, 0.00282, 0.02301),
    (54, 39.01830, -101.02018, 0.00508, 0.02538),
    (55, 11.56594, -85.61392, 0.03955, 0.07593),
    (56, 18.56667, -99.04753, 0.00659, 0.05473),
]


def pw_fault_table():
    df = pd.DataFrame(PW_FAULT, columns=["bus", "I [pu]", "Angle [deg]", "R_th [pu]", "X_th [pu]"])
    return df.set_index("bus")


def fault_zbus(net, loads=True):
    """Zbus seen by a three-phase fault, built from a solved net.

    Generators enter as 1/(R + jX''); with loads=True every load enters as the constant
    admittance S*/|V|^2 at its pre-fault voltage, which is how PowerWorld treats loads.
    Returns (Zbus, pre-fault complex voltages), both in BUSES order.
    """
    order = [b for b, _, _ in BUSES]
    lookup = net._pd2ppc_lookups["bus"]
    ix = [lookup[b] for b in order]
    y = net._ppc["internal"]["Ybus"].toarray()[np.ix_(ix, ix)].copy()
    pos = {b: k for k, b in enumerate(order)}
    on = set(net.gen.bus[net.gen.pv_on.astype(bool)]) | {SLACK_BUS}
    for b in on:
        r, x = GEN_Z[b]
        units = max(1, int((net.gen.bus[net.gen.pv_on.astype(bool)] == b).sum()))
        y[pos[b], pos[b]] += units / complex(r, x)
    vm = net.res_bus.vm_pu[order].to_numpy()
    va = np.deg2rad(net.res_bus.va_degree[order].to_numpy())
    v0 = vm * np.exp(1j * va)
    if loads:
        for i in net.load.index[net.load.in_service]:
            b = net.load.at[i, "bus"]
            s = complex(net.res_load.at[i, "p_mw"], net.res_load.at[i, "q_mvar"]) / S_BASE
            y[pos[b], pos[b]] += np.conj(s) / vm[pos[b]] ** 2
    return np.linalg.inv(y), v0


def fault_results(net, loads=True):
    """Thevenin impedance and fault current at every bus (classical method, V_F = pre-fault voltage)."""
    z, v0 = fault_zbus(net, loads)
    zkk = np.diag(z)
    i_f = v0 / zkk
    return pd.DataFrame({"I [pu]": np.abs(i_f), "Angle [deg]": np.angle(i_f, deg=True),
                         "R_th [pu]": zkk.real, "X_th [pu]": zkk.imag},
                        index=pd.Index([b for b, _, _ in BUSES], name="bus"))


def fault_measure(net):
    """Flat Series of every compared quantity, for residual_band()."""
    r = fault_results(net)
    return pd.Series({f"{c} {b}": r.at[b, c] for b in r.index for c in r.columns})


def calc_sc_currents(gen_cos_phi=0.85):
    """pandapower IEC 60909 maximum three-phase currents [pu]; loads are ignored by the standard."""
    net = build_case()
    tb.replace_ext_grid_by_gen(net, slack=True)
    net.gen["vn_kv"] = net.bus.vn_kv[net.gen.bus].to_numpy()
    net.gen["sn_mva"] = S_BASE
    net.gen["cos_phi"] = gen_cos_phi
    for i in net.gen.index:
        r, x = GEN_Z.get(net.gen.at[i, "bus"], (0.0, 0.05))
        net.gen.at[i, "xdss_pu"] = x
        net.gen.at[i, "rdss_ohm"] = r * net.gen.at[i, "vn_kv"] ** 2 / S_BASE
    sc.calc_sc(net, case="max", fault="3ph")
    return net.res_bus_sc.ikss_ka / (S_BASE / (np.sqrt(3) * net.bus.vn_kv))
