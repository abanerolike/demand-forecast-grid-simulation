"""Five-bus system for the balanced three-phase fault study (Example 8.5).

All impedances are in per unit on S_BASE, read from the PowerWorld case.
Generator subtransient reactances were confirmed from the positive-sequence
Ybus diagonal (Y11 = -j72.22, Y33 = -j144.44).
"""
import numpy as np
import pandas as pd
import pandapower as pp

S_BASE = 100.0                      # MVA
V_PREFAULT = 1.05                   # pu, solved power flow at no load
BUS_KV = {1: 15.0, 2: 345.0, 3: 15.0, 4: 345.0, 5: 345.0}
BUSES = list(BUS_KV)

# (bus, X''d [pu])
GENS = [(1, 0.045), (3, 0.0225)]
# (hv bus, lv bus, X [pu]) - hv bus is the PowerWorld From bus
TRAFOS = [(5, 1, 0.02), (4, 3, 0.01)]
# (from bus, to bus, X [pu]) - R and B are zero in the case
LINES = [(4, 2, 0.10), (5, 2, 0.05), (5, 4, 0.025)]


def conversion_table():
    """Per-unit to pandapower conversion for every branch."""
    rows = []
    for hv, lv, x in TRAFOS:
        rows.append({"Branch": f"T {hv}-{lv}", "X [pu]": x, "Z_base [ohm]": np.nan,
                     "Input": "vk_percent", "Value": 100 * x})
    for f, t, x in LINES:
        zb = BUS_KV[f] ** 2 / S_BASE
        rows.append({"Branch": f"L {f}-{t}", "X [pu]": x, "Z_base [ohm]": zb,
                     "Input": "x_ohm_per_km (length_km=1)", "Value": x * zb})
    return pd.DataFrame(rows)


def build_case(gen_cos_phi=0.85):
    """Build the network. cos_phi only matters for the IEC K_G factor."""
    net = pp.create_empty_network(f_hz=60, sn_mva=S_BASE)
    for b, kv in BUS_KV.items():
        pp.create_bus(net, vn_kv=kv, name=str(b), index=b)
    for b, xdd in GENS:
        pp.create_gen(net, b, p_mw=0.0, vm_pu=V_PREFAULT, slack=(b == 1),
                      sn_mva=S_BASE, vn_kv=BUS_KV[b], xdss_pu=xdd, rdss_ohm=0.0,
                      cos_phi=gen_cos_phi)
    for hv, lv, x in TRAFOS:
        pp.create_transformer_from_parameters(
            net, hv, lv, sn_mva=S_BASE, vn_hv_kv=BUS_KV[hv], vn_lv_kv=BUS_KV[lv],
            vk_percent=100 * x, vkr_percent=0.0, pfe_kw=0.0, i0_percent=0.0)
    for f, t, x in LINES:
        zb = BUS_KV[f] ** 2 / S_BASE
        pp.create_line_from_parameters(net, f, t, length_km=1, r_ohm_per_km=0.0,
                                       x_ohm_per_km=x * zb, c_nf_per_km=0.0, max_i_ka=2.0)
    return net


def ybus_with_generators(net):
    """Positive-sequence Ybus (bus order 1..5) with 1/jX''d added on generator buses."""
    lookup = net._pd2ppc_lookups["bus"]
    ix = [lookup[b] for b in BUSES]
    y = net._ppc["internal"]["Ybus"].toarray()[np.ix_(ix, ix)].copy()
    for b, xdd in GENS:
        k = BUSES.index(b)
        y[k, k] += 1 / (1j * xdd)
    return y


def classical_fault(zbus, k, v_f=V_PREFAULT):
    """Bolted three-phase fault at bus index k: returns I_F and bus voltages."""
    i_f = v_f / zbus[k, k]
    v = v_f - zbus[:, k] * i_f
    return i_f, v


def compare(rows):
    """Comparison table; fails if any row exceeds its tolerance."""
    df = pd.DataFrame(rows, columns=["Quantity", "PowerWorld", "pandapower", "Tolerance", "Source"])
    df.insert(3, "Abs diff", (df["pandapower"] - df["PowerWorld"]).abs())
    bad = df[df["Abs diff"] > df["Tolerance"]]
    assert bad.empty, f"Out of tolerance:\n{bad}"
    return df
