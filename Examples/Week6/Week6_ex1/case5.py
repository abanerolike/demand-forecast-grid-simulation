"""Five-bus system with sequence data for the single line-to-ground fault study (Example 10.8).

Same network as the balanced fault study (Example 8.5) plus negative- and
zero-sequence data. All impedances are in per unit on S_BASE, read from the
PowerWorld case (Fault Analysis > Sequence Data). Both transformers are
delta (generator side) / grounded wye (345 kV side) with zero phase shift,
so the zero-sequence network is cut at the delta winding and the generators
feed zero-sequence current only into a fault on their own bus.
"""
import numpy as np
import pandas as pd
import pandapower as pp

S_BASE = 100.0                      # MVA
V_PREFAULT = 1.05                   # pu, solved power flow at no load
BUS_KV = {1: 15.0, 2: 345.0, 3: 15.0, 4: 345.0, 5: 345.0}
BUSES = list(BUS_KV)

# (bus, X1, X2, X0, Xn) - Xn is the neutral grounding reactance, it enters the
# zero-sequence circuit as 3*Xn. The display shows 0.00 for G3; 0.0025 is the
# value that matches every zero-sequence Thevenin reactance PowerWorld reports
GENS = [(1, 0.045, 0.045, 0.0125, 0.0),
        (3, 0.0225, 0.0225, 0.005, 0.0025)]
# (grounded wye bus, delta bus, X1 = X2, X0) - the wye side is the hv side
TRAFOS = [(5, 1, 0.02, 0.02), (4, 3, 0.01, 0.01)]
# (from bus, to bus, X1 = X2, X0) - R and B are zero in the case
LINES = [(4, 2, 0.10, 0.30), (5, 2, 0.05, 0.15), (5, 4, 0.025, 0.075)]

A_OP = np.exp(2j * np.pi / 3)
A_MAT = np.array([[1, 1, 1], [1, A_OP ** 2, A_OP], [1, A_OP, A_OP ** 2]])


def sequence_table():
    """Sequence reactances of every element, as entered in the model."""
    rows = [{"Element": f"G{b}", "X1 [pu]": x1, "X2 [pu]": x2, "X0 [pu]": x0,
             "Zero-seq path": f"X0 + 3Xn = {x0 + 3 * xn:.4f} to ground at bus {b}"}
            for b, x1, x2, x0, xn in GENS]
    rows += [{"Element": f"T {w}-{d}", "X1 [pu]": x, "X2 [pu]": x, "X0 [pu]": x0,
              "Zero-seq path": f"X0 to ground at bus {w}, open at bus {d} (delta)"}
             for w, d, x, x0 in TRAFOS]
    rows += [{"Element": f"L {f}-{t}", "X1 [pu]": x, "X2 [pu]": x, "X0 [pu]": x0,
              "Zero-seq path": "series"} for f, t, x, x0 in LINES]
    return pd.DataFrame(rows)


def build_case():
    """Positive-sequence pandapower network used for the pre-fault power flow."""
    net = pp.create_empty_network(f_hz=60, sn_mva=S_BASE)
    for b, kv in BUS_KV.items():
        pp.create_bus(net, vn_kv=kv, name=str(b), index=b)
    for b, x1, *_ in GENS:
        pp.create_gen(net, b, p_mw=0.0, vm_pu=V_PREFAULT, slack=(b == 1),
                      sn_mva=S_BASE, vn_kv=BUS_KV[b], xdss_pu=x1, rdss_ohm=0.0)
    for w, d, x, _ in TRAFOS:
        pp.create_transformer_from_parameters(
            net, w, d, sn_mva=S_BASE, vn_hv_kv=BUS_KV[w], vn_lv_kv=BUS_KV[d],
            vk_percent=100 * x, vkr_percent=0.0, pfe_kw=0.0, i0_percent=0.0)
    for f, t, x, _ in LINES:
        zb = BUS_KV[f] ** 2 / S_BASE
        pp.create_line_from_parameters(net, f, t, length_km=1, r_ohm_per_km=0.0,
                                       x_ohm_per_km=x * zb, c_nf_per_km=0.0, max_i_ka=2.0)
    return net


def pandapower_ybus(net):
    """Positive-sequence Ybus from pandapower (bus order 1..5), generators added as 1/jX1."""
    lookup = net._pd2ppc_lookups["bus"]
    ix = [lookup[b] for b in BUSES]
    y = net._ppc["internal"]["Ybus"].toarray()[np.ix_(ix, ix)].copy()
    for b, x1, *_ in GENS:
        y[BUSES.index(b), BUSES.index(b)] += 1 / (1j * x1)
    return y


def _stamp(y, f, t, x):
    """Add a series reactance between two buses (t=None means to ground)."""
    adm = 1 / (1j * x)
    i = BUSES.index(f)
    y[i, i] += adm
    if t is not None:
        j = BUSES.index(t)
        y[j, j] += adm
        y[i, j] -= adm
        y[j, i] -= adm


def sequence_ybus(seq):
    """Hand-built Ybus of sequence 1, 2 or 0, generators included."""
    y = np.zeros((len(BUSES), len(BUSES)), dtype=complex)
    for b, x1, x2, x0, xn in GENS:
        _stamp(y, b, None, {1: x1, 2: x2, 0: x0 + 3 * xn}[seq])
    for w, d, x, x0 in TRAFOS:
        if seq == 0:
            _stamp(y, w, None, x0)          # delta winding: path to ground on the wye side only
        else:
            _stamp(y, w, d, x)
    for f, t, x, x0 in LINES:
        _stamp(y, f, t, x0 if seq == 0 else x)
    return y


def slg_fault(z1, z2, z0, k, v_f=V_PREFAULT):
    """Bolted phase-a-to-ground fault at bus index k.

    Returns the sequence fault current I0 = I1 = I2 and the sequence bus
    voltages as a (3, n) array in the order (0, 1, 2).
    """
    i_seq = v_f / (z0[k, k] + z1[k, k] + z2[k, k])
    v0 = -z0[:, k] * i_seq
    v1 = v_f - z1[:, k] * i_seq
    v2 = -z2[:, k] * i_seq
    return i_seq, np.vstack([v0, v1, v2])


def to_phase(seq012):
    """Sequence components (0, 1, 2) along axis 0 -> phase components (a, b, c)."""
    return A_MAT @ seq012


def compare(rows):
    """Comparison table; fails if any row exceeds its tolerance."""
    df = pd.DataFrame(rows, columns=["Quantity", "PowerWorld", "pandapower", "Tolerance", "Source"])
    df.insert(3, "Abs diff", (df["pandapower"] - df["PowerWorld"]).abs())
    bad = df[df["Abs diff"] > df["Tolerance"]]
    assert bad.empty, f"Out of tolerance:\n{bad}"
    return df


C_MAX = 1.1                         # IEC 60909 voltage factor, max case


def k_t(x):
    """IEC 60909 network transformer correction for a reactance x [pu on its own rating]."""
    return 0.95 * C_MAX / (1 + 0.6 * x)


def build_sc_case(gen_grounding="none", gen_cos_phi=0.85):
    """pandapower network with zero-sequence data for calc_sc(fault="1ph").

    calc_sc models every generator as ungrounded in the zero sequence. With
    gen_grounding="trafo" each generator's X0 + 3Xn is added as a YNd
    transformer to an empty delta-side bus (a grounding bank); "compensated"
    divides vk0 by the transformer's own K_T so the path enters at X0 + 3Xn.
    """
    net = build_case()
    net.gen["cos_phi"] = gen_cos_phi
    zero = {"vector_group": "YNd", "vkr0_percent": 0.0, "mag0_percent": 100.0,
            "mag0_rx": 0.0, "si0_hv_partial": 0.9}
    for i, (w, d, x, x0) in enumerate(TRAFOS):
        for key, val in zero.items():
            net.trafo.loc[i, key] = val
        net.trafo.loc[i, "vk0_percent"] = 100 * x0
    for i, (f, t, x, x0) in enumerate(LINES):
        net.line.loc[i, ["r0_ohm_per_km", "x0_ohm_per_km", "c0_nf_per_km"]] = \
            [0.0, x0 * BUS_KV[f] ** 2 / S_BASE, 0.0]
    if gen_grounding in ("trafo", "compensated"):
        for b, x1, x2, x0, xn in GENS:
            z0 = x0 + 3 * xn
            scale = k_t(z0) if gen_grounding == "compensated" else 1.0
            dummy = pp.create_bus(net, vn_kv=BUS_KV[b], name=f"ground bank {b}")
            pp.create_transformer_from_parameters(
                net, b, dummy, sn_mva=S_BASE, vn_hv_kv=BUS_KV[b], vn_lv_kv=BUS_KV[b],
                vk_percent=100 * z0, vkr_percent=0.0, pfe_kw=0.0, i0_percent=0.0,
                vk0_percent=100 * z0 / scale, **zero)
    return net
