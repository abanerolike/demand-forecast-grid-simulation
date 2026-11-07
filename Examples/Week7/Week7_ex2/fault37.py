"""Sequence data and single line-to-ground fault helpers for the 37-bus design case (Problem 10.64).

The network comes from case37.py; this module adds the negative- and zero-sequence data read
from PowerWorld (Fault Analysis > Sequence Data, saved with Save Data) and the PowerWorld fault
results. Modelling rules that reproduce PowerWorld, found by matching every Thevenin impedance:
  - negative sequence: same network as the positive sequence, generators at R2 + jX2, loads at
    their positive-sequence constant admittance (the case leaves the negative-sequence load
    fields at zero, which means "same as positive")
  - zero sequence: lines at R0 + jX0 with their zero-sequence charging; loads are not grounded;
    switched shunts carry no zero-sequence Mvar; every transformer is "Unknown" configuration and
    is left out entirely (no series path, no magnetizing branch); generators at R0 + j(X0 + 3Xn)
"""
import numpy as np
import pandas as pd

from case37 import S_BASE, SLACK_BUS, BUSES

ORDER = [b for b, _, _ in BUSES]
POS = {b: k for k, b in enumerate(ORDER)}
KV = {b: kv for b, _, kv in BUSES}
A_OP = np.exp(2j * np.pi / 3)
A_MAT = np.array([[1, 1, 1], [1, A_OP ** 2, A_OP], [1, A_OP, A_OP ** 2]])

# Generator sequence impedances [pu on 100 MVA]: bus, id, R1, X1, R2, X2, R0, X0, Rn, Xn
#    (REDBUD69 unit 1 is out of service)
GEN_SEQ = [
    (14, '1', 0.00163, 0.3, 0.00163, 0.32, 0.00163, 0.1, 0.0, 0.0),
    (28, '1', 0.0, 0.1, 0.0, 0.11, 0.0, 0.035, 0.0, 0.0),
    (28, '2', 0.0, 0.1, 0.0, 0.11, 0.0, 0.035, 0.0, 0.0),
    (31, '1', 0.0, 0.07, 0.0, 0.07, 0.0, 0.03, 0.0, 0.0),
    (44, '1', 0.004, 0.11, 0.004, 0.12, 0.004, 0.03, 0.0, 0.0),
    (48, '1', 0.00156, 0.25, 0.00156, 0.27, 0.00156, 0.1, 0.0, 0.0),
    (50, '1', 0.0035, 0.25, 0.0035, 0.25, 0.0035, 0.125, 0.0, 0.0),
    (53, '1', 0.00132, 0.12, 0.00132, 0.12, 0.00132, 0.04, 0.0, 0.0),
    (54, '1', 0.00112, 0.15, 0.00112, 0.16, 0.00112, 0.06, 0.0, 0.0),
]

# Branch zero-sequence data: from, to, circuit, R0, X0, B0 [pu], transformer with "Unknown" configuration
BRANCH_ZERO = [
    (1, 31, '1', 0.002925, 0.0425, 0.182, False),
    (1, 40, '1', 0.001, 0.0623, 0.0, True),
    (3, 40, '1', 0.012, 0.092, 0.0182, False),
    (3, 41, '1', 0.02255, 0.14275, 0.0135, False),
    (5, 18, '1', 0.078325, 0.191875, 0.0015, False),
    (5, 44, '1', 0.086575, 0.206325, 0.0016, False),
    (10, 13, '1', 0.084375, 0.19725, 0.0012, False),
    (10, 19, '1', 0.10125, 0.23825, 0.0021, False),
    (10, 39, '1', 0.00106, 0.03949, -0.0041, True),
    (12, 17, '1', 0.0543, 0.173375, 0.0019, False),
    (12, 18, '1', 0.068675, 0.222725, 0.0019, False),
    (12, 27, '1', 0.0355, 0.188925, 0.0695, False),
    (12, 40, '1', 0.00107, 0.04039, -0.0042, True),
    (12, 40, '2', 0.00107, 0.04039, -0.0042, True),
    (13, 55, '1', 0.0554, 0.226, 0.0017, False),
    (14, 34, '1', 0.065625, 0.160725, 0.0012, False),
    (14, 44, '1', 0.105275, 0.213625, 0.0474, False),
    (15, 16, '1', 0.018825, 0.06455, 0.0206, False),
    (15, 24, '1', 0.045875, 0.071125, 0.0221, False),
    (15, 54, '1', 0.02115, 0.011625, 0.021, False),
    (15, 54, '2', 0.021375, 0.01175, 0.0213, False),
    (15, 54, '3', 0.021475, 0.0118, 0.0214, False),
    (16, 27, '1', 0.00915, 0.0328, 0.046, False),
    (17, 19, '1', 0.067575, 0.090325, 0.002, False),
    (18, 37, '1', 0.025425, 0.013975, 0.0253, False),
    (18, 37, '2', 0.025425, 0.013975, 0.0253, False),
    (20, 34, '1', 0.060575, 0.14655, 0.0011, False),
    (20, 48, '1', 0.053325, 0.13025, 0.0009, False),
    (20, 50, '1', 0.106, 0.187725, 0.0008, False),
    (21, 48, '1', 0.0456, 0.1059, 0.0075, False),
    (21, 48, '2', 0.045725, 0.10615, 0.0078, False),
    (24, 44, '1', 0.099825, 0.249125, 0.002, False),
    (28, 29, '1', 0.00087, 0.051, 0.0, True),
    (28, 29, '2', 0.00087, 0.051, 0.0, True),
    (29, 41, '1', 0.0467, 0.31475, 0.0356, False),
    (30, 32, '1', 0.005625, 0.0335, 0.004, False),
    (30, 41, '1', 0.018375, 0.109875, 0.0123, False),
    (31, 28, '1', 0.0056, 0.0817, 0.35, False),
    (31, 38, '1', 0.001875, 0.0273, 0.117, False),
    (32, 29, '1', 0.02575, 0.142025, 0.0164, False),
    (33, 32, '1', 0.0025, 0.0723, 0.0, True),
    (33, 50, '1', 0.1255, 0.2525, 0.0025, False),
    (35, 31, '1', 0.00087, 0.051, 0.0, True),
    (35, 39, '1', 0.0257, 0.181325, 0.0176, False),
    (35, 56, '1', 0.031075, 0.196175, 0.0192, False),
    (39, 38, '1', 0.00094, 0.05107, -0.0127, True),
    (39, 38, '2', 0.00095, 0.05116, -0.0124, True),
    (39, 40, '1', 0.029025, 0.202125, 0.0225, False),
    (39, 47, '1', 0.019375, 0.1311, 0.0154, False),
    (44, 41, '1', 0.0025, 0.07144, -0.0045, True),
    (44, 41, '2', 0.00244, 0.06829, -0.006, True),
    (47, 53, '1', 0.0051, 0.0173, 0.1601, False),
    (48, 47, '1', 0.00101, 0.03925, -0.0035, True),
    (48, 54, '1', 0.036825, 0.09, 0.0744, False),
    (54, 53, '1', 0.00134, 0.04988, -0.0004, True),
    (54, 55, '1', 0.15615, 0.20615, 0.0003, False),
    (56, 29, '1', 0.019275, 0.136, 0.0138, False),
]

# PowerWorld Fault Definitions, bolted fault at each bus (pre-fault profile = solved power flow):
#    bus, Fault 1 (3PB) |I| [pu], angle [deg], Thevenin R, X [pu],
#         Fault 2 (SLG) |I| [pu], angle [deg], Thevenin R, X [pu] (= Z1 + Z2 + Z0)
PW_FAULT = [
    (1, 28.09167, -100.74102, 0.00324, 0.03631, 21.98924, -101.78707, 0.00989, 0.13935),
    (3, 19.09824, -101.04310, 0.00832, 0.05179, 5.63511, -101.64816, 0.07905, 0.52742),
    (5, 11.49937, -93.71832, 0.03028, 0.08190, 8.27169, -93.73667, 0.12617, 0.34160),
    (10, 17.68211, -99.58422, 0.01059, 0.05681, 6.04822, -90.49500, 0.17044, 0.47734),
    (12, 21.99636, -99.46339, 0.00974, 0.04536, 10.28450, -96.71999, 0.07633, 0.28771),
    (13, 10.52769, -91.50179, 0.03323, 0.08902, 5.51170, -89.76945, 0.20577, 0.50409),
    (14, 11.82551, -91.64812, 0.03197, 0.07855, 8.86637, -91.11921, 0.13080, 0.31312),
    (15, 30.10966, -97.15185, 0.00880, 0.03224, 27.25943, -97.71584, 0.02812, 0.10711),
    (16, 20.46120, -97.19516, 0.01341, 0.04719, 15.12132, -97.43524, 0.05362, 0.19178),
    (17, 11.87118, -93.82928, 0.02698, 0.08089, 6.26876, -94.03270, 0.15167, 0.46010),
    (18, 11.09742, -94.10935, 0.03177, 0.08404, 6.97909, -95.11752, 0.14447, 0.40351),
    (19, 11.06183, -91.90219, 0.03128, 0.08603, 5.63990, -91.10314, 0.19110, 0.50360),
    (20, 17.35237, -94.01681, 0.01876, 0.05513, 13.74025, -93.89258, 0.07154, 0.20872),
    (21, 16.88152, -95.21182, 0.01819, 0.05712, 13.22887, -95.70709, 0.06773, 0.21925),
    (24, 19.44678, -91.15467, 0.01907, 0.04790, 15.37712, -90.16277, 0.07550, 0.18046),
    (27, 18.53527, -97.74427, 0.01424, 0.05235, 12.89602, -97.54426, 0.06220, 0.22553),
    (28, 40.67445, -99.37981, 0.00151, 0.02528, 46.01390, -99.98003, 0.00331, 0.06707),
    (29, 28.94295, -100.09696, 0.00328, 0.03524, 5.44797, -97.19281, 0.08062, 0.55836),
    (30, 19.28990, -99.39022, 0.00878, 0.05153, 4.80490, -100.14844, 0.09750, 0.62192),
    (31, 41.08418, -100.02402, 0.00208, 0.02498, 41.75648, -101.23355, 0.00458, 0.07386),
    (32, 19.89226, -98.90391, 0.00851, 0.05012, 4.81474, -99.62491, 0.09768, 0.62251),
    (33, 11.92963, -98.41867, 0.01675, 0.08353, 5.62474, -91.28326, 0.17177, 0.51414),
    (34, 11.88991, -91.87094, 0.03179, 0.07797, 8.78286, -92.12620, 0.12772, 0.31723),
    (35, 22.41527, -100.41632, 0.00452, 0.04550, 6.84521, -98.34251, 0.06055, 0.44504),
    (37, 10.58851, -91.78439, 0.03682, 0.08647, 6.66593, -92.89167, 0.16747, 0.41538),
    (38, 33.31112, -100.33553, 0.00290, 0.03062, 27.24363, -101.62070, 0.00812, 0.11254),
    (39, 31.54956, -100.04066, 0.00449, 0.03183, 11.59934, -99.98223, 0.03694, 0.25972),
    (40, 25.68456, -100.31528, 0.00651, 0.03850, 6.66293, -101.32034, 0.06741, 0.44654),
    (41, 25.13284, -101.30921, 0.00617, 0.03942, 5.35193, -101.55616, 0.08449, 0.55567),
    (44, 26.82610, -100.68888, 0.00779, 0.03722, 29.38290, -101.25426, 0.02033, 0.10214),
    (47, 30.39805, -101.47351, 0.00509, 0.03242, 23.96317, -102.55811, 0.01701, 0.12373),
    (48, 27.04503, -99.29334, 0.00860, 0.03664, 23.88440, -100.91586, 0.02567, 0.12525),
    (50, 13.82483, -95.50580, 0.01965, 0.07111, 12.63727, -98.59395, 0.05183, 0.23653),
    (53, 30.28212, -101.72417, 0.00498, 0.03267, 28.04492, -104.21890, 0.01151, 0.10645),
    (54, 30.94403, -99.27943, 0.00737, 0.03180, 28.70560, -101.53731, 0.01978, 0.10368),
    (55, 10.87079, -86.07636, 0.04142, 0.08112, 6.92062, -84.24978, 0.20728, 0.37586),
    (56, 16.71004, -98.57515, 0.00784, 0.06074, 5.24388, -97.59338, 0.08487, 0.57928),
]


# PowerWorld Single Fault, SLG at ORANGE69 (bus 16): phase voltages during the fault
#    bus, |Va|, |Vb|, |Vc| [pu], angle a, b, c [deg]
PW_ORANGE_V = [
    (16, 0.00000, 1.15884, 1.15272, -136.72, -153.91, 108.01),
    (27, 0.05992, 1.15764, 1.15532, -23.53, -153.81, 107.94),
    (12, 0.40603, 1.14585, 1.17522, -18.44, -152.38, 107.95),
    (17, 0.44619, 1.13381, 1.15139, -20.30, -152.32, 106.99),
    (19, 0.47339, 1.13077, 1.14347, -21.09, -151.47, 107.13),
    (15, 0.49599, 1.04266, 1.03163, -22.77, -144.72, 100.20),
    (55, 0.52432, 1.04890, 1.03793, -26.49, -147.35, 101.47),
    (13, 0.52981, 1.08681, 1.07848, -23.54, -148.41, 104.74),
    (37, 0.53400, 1.07226, 1.08055, -24.53, -151.55, 101.48),
    (18, 0.53555, 1.07396, 1.08235, -24.49, -151.49, 101.49),
    (54, 0.53566, 1.05228, 1.01336, -26.72, -143.44, 100.05),
    (10, 0.54727, 1.12333, 1.13877, -19.74, -148.76, 107.80),
    (24, 0.57107, 1.02985, 1.01286, -24.84, -143.97, 98.77),
    (5, 0.67592, 1.03078, 1.03693, -24.46, -146.52, 98.25),
    (21, 0.71559, 1.02013, 0.98374, -26.24, -140.91, 96.25),
    (48, 0.72016, 1.02543, 0.99004, -25.80, -140.54, 96.59),
    (20, 0.77923, 1.00647, 0.98500, -25.07, -140.60, 95.60),
    (34, 0.78465, 0.99446, 0.97642, -26.12, -141.76, 94.17),
    (14, 0.79998, 0.99179, 0.98012, -25.52, -141.44, 94.18),
    (40, 0.80035, 0.96195, 0.95614, -19.47, -134.51, 94.81),
    (47, 0.80275, 0.95470, 0.95587, -20.51, -135.28, 94.41),
    (53, 0.80450, 0.95800, 0.95838, -20.45, -135.24, 94.40),
    (3, 0.82001, 0.96369, 0.96102, -19.99, -135.38, 95.06),
    (44, 0.83390, 1.00188, 0.99944, -23.89, -140.04, 95.06),
    (39, 0.85021, 0.98016, 0.97650, -17.83, -133.80, 97.70),
    (41, 0.85290, 0.96904, 0.97183, -20.43, -136.33, 95.80),
    (33, 0.86896, 1.00326, 0.98892, -21.03, -137.12, 98.00),
    (50, 0.87030, 1.00193, 0.99552, -22.87, -138.35, 96.62),
    (30, 0.87364, 0.97854, 0.97977, -19.17, -135.60, 97.39),
    (32, 0.88114, 0.98268, 0.98358, -18.63, -135.20, 98.04),
    (1, 0.88718, 0.99878, 0.98883, -15.15, -132.24, 100.78),
    (35, 0.89223, 0.99862, 0.99278, -15.70, -132.65, 100.59),
    (38, 0.89224, 0.99954, 0.99141, -15.19, -132.28, 100.96),
    (56, 0.90384, 0.99963, 0.99513, -15.64, -132.83, 101.05),
    (31, 0.91113, 1.00810, 0.99943, -14.19, -131.67, 102.31),
    (29, 0.91375, 1.00243, 0.99881, -15.18, -132.55, 101.78),
    (28, 0.94020, 1.01451, 1.00730, -12.34, -130.44, 104.98),
]

# PowerWorld Single Fault, SLG at ORANGE69: generator phase currents [pu], angles [deg]
#    (both ELM345 units carry the same current)
PW_ORANGE_GEN = [
    (50, "1", 0.89958, 0.22947, 0.48400, -73.37, -152.86, 80.22),
    (48, "1", 1.59077, 0.66250, 0.32881, -94.11, -87.47, -100.14),
    (28, "1", 1.62484, 1.08842, 1.82378, -44.40, -142.50, 99.38),
    (28, "2", 1.62484, 1.08842, 1.82378, -44.40, -142.50, 99.38),
    (53, "1", 2.52521, 1.09060, 2.28371, -76.09, 168.65, 78.33),
    (44, "1", 2.71686, 0.33590, 0.47731, -96.88, -124.36, -19.16),
    (31, "1", 2.99968, 1.60927, 3.02728, -62.88, -167.41, 86.15),
    (54, "1", 5.34986, 2.81837, 1.25518, -95.67, -115.40, -102.01),
]
PW_ORANGE_IF = (15.121, -97.44)     # |I_F| [pu], angle [deg] from the Single Fault dialog


def online_units(net):
    """In-service generator units with their pre-fault complex power [pu]."""
    units = []
    for bus, gid, *z in GEN_SEQ:
        name = f"G{bus}-{gid}"
        if bus == SLACK_BUS:
            row = net.res_ext_grid.loc[net.ext_grid.index[net.ext_grid.name == name][0]]
        else:
            i = net.gen.index[net.gen.name == name][0]
            if not net.gen.at[i, "pv_on"]:
                continue
            if i in net.parked:
                row = net.res_sgen.loc[net.sgen.index[net.sgen.name == f"park{i}"][0]]
            else:
                row = net.res_gen.loc[i]
        units.append((bus, gid, complex(row.p_mw, row.q_mvar) / S_BASE, z))
    return units


def sequence_zbus(net, unknown="open"):
    """Positive-, negative- and zero-sequence Zbus (ORDER) and pre-fault voltages from a solved net.

    unknown sets how an "Unknown" transformer enters the zero sequence: "open" (PowerWorld),
    "series" (grounded wye on both sides: R0 + jX0 between the buses) or "ground_hv" (delta on the
    low-voltage side, grounded wye on the high-voltage side: R0 + jX0 to ground at the hv bus).
    """
    lookup = net._pd2ppc_lookups["bus"]
    ix = [lookup[b] for b in ORDER]
    y_net = net._ppc["internal"]["Ybus"].toarray()[np.ix_(ix, ix)]
    vm = net.res_bus.vm_pu[ORDER].to_numpy()
    v0 = vm * np.exp(1j * np.deg2rad(net.res_bus.va_degree[ORDER].to_numpy()))
    y_load = np.zeros(len(ORDER), complex)
    for i in net.load.index[net.load.in_service]:
        b = net.load.at[i, "bus"]
        s = complex(net.res_load.at[i, "p_mw"], net.res_load.at[i, "q_mvar"]) / S_BASE
        y_load[POS[b]] += np.conj(s) / vm[POS[b]] ** 2
    y1, y2 = y_net + np.diag(y_load), y_net + np.diag(y_load)
    y0 = np.zeros_like(y1)
    for f, t, ckt, r0, x0, b0, xf in BRANCH_ZERO:
        i, j, y = POS[f], POS[t], 1 / complex(r0, x0)
        if xf:
            if unknown == "series":
                y0[i, i] += y; y0[j, j] += y; y0[i, j] -= y; y0[j, i] -= y
            elif unknown == "ground_hv":
                hv = f if KV[f] > KV[t] else t
                y0[POS[hv], POS[hv]] += y
            continue
        y0[i, i] += y + 0.5j * b0; y0[j, j] += y + 0.5j * b0
        y0[i, j] -= y; y0[j, i] -= y
    for bus, gid, s, (r1, x1, r2, x2, r0, x0, rn, xn) in online_units(net):
        k = POS[bus]
        y1[k, k] += 1 / complex(r1, x1)
        y2[k, k] += 1 / complex(r2, x2)
        y0[k, k] += 1 / complex(r0 + 3 * rn, x0 + 3 * xn)
    return np.linalg.inv(y1), np.linalg.inv(y2), np.linalg.inv(y0), v0


def slg_results(net, unknown="open"):
    """Bolted SLG fault at every bus: series impedance Z1 + Z2 + Z0 and fault current 3*I0."""
    z1, z2, z0, v0 = sequence_zbus(net, unknown)
    zs = np.diag(z1) + np.diag(z2) + np.diag(z0)
    i_f = 3 * v0 / zs
    return pd.DataFrame({"I [pu]": np.abs(i_f), "Angle [deg]": np.angle(i_f, deg=True),
                         "R_th [pu]": zs.real, "X_th [pu]": zs.imag,
                         "R1 [pu]": np.diag(z1).real, "X1 [pu]": np.diag(z1).imag,
                         "Z0 [pu]": np.abs(np.diag(z0))},
                        index=pd.Index(ORDER, name="bus"))


def slg_at(net, bus, unknown="open"):
    """SLG fault (phase a) at one bus: fault current, phase voltages at every bus, generator currents."""
    z1, z2, z0, v0 = sequence_zbus(net, unknown)
    k = POS[bus]
    i0 = v0[k] / (z1[k, k] + z2[k, k] + z0[k, k])
    v012 = np.vstack([-z0[:, k] * i0, v0 - z1[:, k] * i0, -z2[:, k] * i0])
    vabc = A_MAT @ v012
    gens = {}
    for b, gid, s, (r1, x1, r2, x2, r0, x0, rn, xn) in online_units(net):
        p = POS[b]
        i_pre = np.conj(s / v0[p])                         # pre-fault unit current
        i012 = np.array([-v012[0, p] / complex(r0 + 3 * rn, x0 + 3 * xn),
                         i_pre + (v0[p] - v012[1, p]) / complex(r1, x1),   # source behind R1 + jX1
                         -v012[2, p] / complex(r2, x2)])
        gens[(b, gid)] = A_MAT @ i012
    return 3 * i0, pd.DataFrame(vabc.T, index=pd.Index(ORDER, name="bus"), columns=list("abc")), gens


def fault_measure(net):
    """Flat Series of every compared quantity, for residual_band()."""
    r = slg_results(net)
    cols = ("I [pu]", "Angle [deg]", "R_th [pu]", "X_th [pu]", "R1 [pu]", "X1 [pu]")
    vals = {f"{c} {b}": r.at[b, c] for b in r.index for c in cols}
    for b in ORDER:
        vals[f"V0 {b}"] = net.res_bus.vm_pu[b]
        vals[f"delta0 {b}"] = net.res_bus.va_degree[b]
    i_f, v, gens = slg_at(net, 16)
    for b in v.index:
        for ph in "abc":
            vals[f"V{ph} {b}"] = abs(v.at[b, ph])
            vals[f"ang{ph} {b}"] = np.angle(v.at[b, ph], deg=True)
    for (b, gid), i in gens.items():
        for ph, val in zip("abc", i):
            vals[f"I{ph} G{b}-{gid}"] = abs(val)
            vals[f"angI{ph} G{b}-{gid}"] = np.angle(val, deg=True)
    vals["I_F ORANGE69"] = abs(i_f)
    vals["I_F angle ORANGE69"] = np.angle(i_f, deg=True)
    return pd.Series(vals)


C_MAX = 1.1                         # IEC 60909 voltage factor, max case


def k_t(x):
    """IEC 60909 network transformer correction for a reactance x [pu on its own rating]."""
    return 0.95 * C_MAX / (1 + 0.6 * x)


def build_sc_case(banks=True, gen_cos_phi=0.85):
    """pandapower network with zero-sequence data for calc_sc(fault="1ph").

    Transformers get vector group "Dd", which blocks the zero sequence on both sides exactly like
    PowerWorld's "Unknown". calc_sc models every generator as ungrounded, so with banks=True each
    unit's R0 + j(X0 + 3Xn) is added as a YNd grounding bank to an empty bus, with vk0 divided by
    the bank's own K_T so the path enters at its data value.
    """
    import pandapower as pp
    import pandapower.toolbox as tb
    from case37 import build_case

    net = build_case()
    tb.replace_ext_grid_by_gen(net, slack=True)
    net.gen["vn_kv"] = net.bus.vn_kv[net.gen.bus].to_numpy()
    net.gen["sn_mva"] = S_BASE
    net.gen["cos_phi"] = gen_cos_phi
    seq = {f"G{b}-{gid}": z for b, gid, *z in GEN_SEQ}
    for i in net.gen.index:
        name = net.gen.at[i, "name"] if net.gen.at[i, "name"] in seq else f"G{SLACK_BUS}-1"
        r1, x1 = seq[name][:2]
        net.gen.at[i, "xdss_pu"] = x1
        net.gen.at[i, "rdss_ohm"] = r1 * net.gen.at[i, "vn_kv"] ** 2 / S_BASE
    zero = {(f, t, ckt): (r0, x0, b0) for f, t, ckt, r0, x0, b0, _ in BRANCH_ZERO}
    for i in net.line.index:
        f, t, ckt = net.line.at[i, "name"].split("-")
        r0, x0, b0 = zero[(int(f), int(t), ckt)]
        zb = KV[int(f)] ** 2 / S_BASE
        net.line.loc[i, ["r0_ohm_per_km", "x0_ohm_per_km", "c0_nf_per_km"]] = \
            [r0 * zb, x0 * zb, b0 / zb / (2 * np.pi * 60) * 1e9]
    bank = {"vector_group": "YNd", "mag0_percent": 100.0, "mag0_rx": 0.0, "si0_hv_partial": 0.9}
    for key, val in bank.items():
        net.trafo[key] = val
    net.trafo["vector_group"] = "Dd"
    net.trafo["vk0_percent"] = net.trafo.vk_percent
    net.trafo["vkr0_percent"] = net.trafo.vkr_percent
    if banks:
        for b, gid, r1, x1, r2, x2, r0, x0, rn, xn in GEN_SEQ:
            if not net.gen.in_service[net.gen.name == f"G{b}-{gid}"].any() and b != SLACK_BUS:
                continue
            r, x = r0 + 3 * rn, x0 + 3 * xn
            dummy = pp.create_bus(net, vn_kv=KV[b], name=f"ground bank {b}-{gid}")
            pp.create_transformer_from_parameters(
                net, b, dummy, sn_mva=S_BASE, vn_hv_kv=KV[b], vn_lv_kv=KV[b],
                vk_percent=100 * np.hypot(r, x), vkr_percent=100 * r, pfe_kw=0.0, i0_percent=0.0,
                vk0_percent=100 * np.hypot(r, x) / k_t(x), vkr0_percent=100 * r / k_t(x), **bank)
    return net
