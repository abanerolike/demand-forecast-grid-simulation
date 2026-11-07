"""Five-bus, two-generator system of Examples 6.9 and 6.14, used again in Example 12.8.

Network data on S_BASE as in Week4_ex1 (Tables 6.1-6.3), with the 200 Mvar shunt
capacitor of Example 6.14 at bus 2. For transient stability both generators use
the classical model; PowerWorld stores X'd and H on each machine's own MVA base
(400 and 800 MVA), so they are converted to S_BASE before they enter Ybus.
"""
import math

import numpy as np
import pandapower as pp

S_BASE = 100.0
F_HZ = 60.0
BUS_KV = {1: 15.0, 2: 345.0, 3: 15.0, 4: 345.0, 5: 345.0}
LINES = [(2, 4, 0.0090, 0.100, 1.72), (2, 5, 0.0045, 0.050, 0.88), (4, 5, 0.00225, 0.025, 0.44)]
TRAFOS = [(5, 1, 0.0015, 0.02), (4, 3, 0.00075, 0.01)]     # hv bus, lv bus, R, X
CAP_MVAR = 200.0                                           # nominal, at 1.0 pu
# GENCLS data as entered in PowerWorld, on the machine's own MVA base
GENS = {1: {"mva": 400.0, "H": 5.0, "D": 1.0, "Xdp": 0.20},
        3: {"mva": 800.0, "H": 5.0, "D": 1.0, "Xdp": 0.20}}


def on_system_base(g):
    """Impedance scales with S_BASE / S_machine, stored energy (H) with S_machine / S_BASE."""
    return {"Xdp": g["Xdp"] * S_BASE / g["mva"], "H": g["H"] * g["mva"] / S_BASE}


def build_case():
    net = pp.create_empty_network(sn_mva=S_BASE, f_hz=F_HZ)
    for b, kv in BUS_KV.items():
        pp.create_bus(net, vn_kv=kv, name=f"Bus {b}", index=b)
    pp.create_ext_grid(net, 1, vm_pu=1.0, va_degree=0.0, name="Gen 1 (slack)")
    pp.create_gen(net, 3, p_mw=520.0, vm_pu=1.05, min_q_mvar=-280.0, max_q_mvar=400.0, name="Gen 3")
    pp.create_load(net, 2, p_mw=800.0, q_mvar=280.0, name="Load 2")
    pp.create_load(net, 3, p_mw=80.0, q_mvar=40.0, name="Load 3")
    pp.create_shunt(net, 2, q_mvar=-CAP_MVAR, p_mw=0.0, name="Cap 2")
    for f, t, r, x, b in LINES:
        zb = BUS_KV[f] ** 2 / S_BASE
        pp.create_line_from_parameters(
            net, f, t, length_km=1.0, r_ohm_per_km=r * zb, x_ohm_per_km=x * zb,
            c_nf_per_km=b / zb / (2 * math.pi * F_HZ) * 1e9, max_i_ka=10.0, name=f"{f}-{t}")
    for hv, lv, r, x in TRAFOS:
        pp.create_transformer_from_parameters(
            net, hv, lv, sn_mva=S_BASE, vn_hv_kv=BUS_KV[hv], vn_lv_kv=BUS_KV[lv],
            vkr_percent=r * 100, vk_percent=math.hypot(r, x) * 100,
            pfe_kw=0.0, i0_percent=0.0, shift_degree=0.0, name=f"{lv}-{hv}")
    return net


def solve(net):
    pp.runpp(net, enforce_q_lims=True, tolerance_mva=1e-10, numba=False)
    return net


def stability_ybus(net, xdp):
    """Y11, Y12, Y22 of equation (12.5.5): network Ybus (shunt capacitor included) plus every load
    as a constant admittance conj(S)/|V|^2 at its pre-fault voltage, plus 1/jX'd at each generator bus.

    `xdp` maps generator bus -> X'd on S_BASE. Rows and columns follow bus numbers 1..5, then the
    internal generator nodes in the order of `xdp`.
    """
    lookup = net._pd2ppc_lookups["bus"]
    buses = sorted(BUS_KV)
    idx = [lookup[b] for b in buses]
    y11 = net._ppc["internal"]["Ybus"].toarray()[np.ix_(idx, idx)].copy()
    for _, ld in net.load.iterrows():
        s = (net.res_load.p_mw[ld.name] + 1j * net.res_load.q_mvar[ld.name]) / S_BASE
        k = buses.index(ld.bus)
        y11[k, k] += np.conj(s) / net.res_bus.vm_pu[ld.bus] ** 2
    gens = list(xdp)
    y12 = np.zeros((len(buses), len(gens)), dtype=complex)
    y22 = np.zeros((len(gens), len(gens)), dtype=complex)
    for j, g in enumerate(gens):
        yg = 1 / (1j * xdp[g])
        k = buses.index(g)
        y11[k, k] += yg
        y12[k, j] = -yg
        y22[j, j] = yg
    return y11, y12, y22
