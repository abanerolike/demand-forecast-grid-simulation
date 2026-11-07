"""Single machine connected to an infinite bus (Examples 12.3-12.7).

Bus 4 is the generator terminal, the transformer (X = 0.10) ties it to bus 1,
and two parallel paths (line 1-2, and lines 1-3 + 3-2) reach the infinite bus 2.
All lines and the transformer are pure reactances in per unit on S_BASE. The
generator uses the classical model (GENCLS): a constant EMF behind X'd.

pandapower has no time-domain solver, so this module only uses pandapower for
the steady-state pieces (power flow, network admittance matrix) and integrates
the swing equation with scipy.
"""
import math

import numpy as np
import pandas as pd
import pandapower as pp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

S_BASE = 100.0          # MVA, Assumed MVA Per Unit Base of the case
TOL_MVA = 0.1           # MVA convergence tolerance of the case
F_HZ = 60.0
OMEGA_SYN = 2 * math.pi * F_HZ
BUS_KV = {1: 138.0, 2: 138.0, 3: 138.0, 4: 13.8}
LINES = [(1, 2, 0.20), (1, 3, 0.10), (2, 3, 0.20)]     # from, to, X (pu)
XTR = 0.10              # transformer 4-1
GEN = {"bus": 4, "P": 1.0, "H": 3.0, "D": 0.0, "Xdp": 0.30}   # pu on S_BASE (Gen MVA Base = S_BASE)


def build_case(vset, lines_out=()):
    """Pre-fault network; `lines_out` holds (from, to) pairs switched out of service."""
    net = pp.create_empty_network(f_hz=F_HZ, sn_mva=S_BASE)
    for b, kv in BUS_KV.items():
        pp.create_bus(net, vn_kv=kv, name=f"Bus {b}", index=b)
    pp.create_ext_grid(net, 2, vm_pu=1.0, va_degree=0.0, name="Infinite bus")
    pp.create_gen(net, GEN["bus"], p_mw=GEN["P"] * S_BASE, vm_pu=vset, name="Gen 4")
    pp.create_transformer_from_parameters(
        net, 1, 4, sn_mva=S_BASE, vn_hv_kv=BUS_KV[1], vn_lv_kv=BUS_KV[4],
        vk_percent=100 * XTR, vkr_percent=0.0, pfe_kw=0.0, i0_percent=0.0, name="T 4-1")
    for f, t, x in LINES:
        zb = BUS_KV[f] ** 2 / S_BASE
        pp.create_line_from_parameters(
            net, f, t, length_km=1.0, r_ohm_per_km=0.0, x_ohm_per_km=x * zb,
            c_nf_per_km=0.0, max_i_ka=1.0, name=f"{f}-{t}",
            in_service=(f, t) not in lines_out)
    return net


def solve(net):
    pp.runpp(net, tolerance_mva=1e-10, numba=False)
    return net


def internal_emf(net):
    """E' = V_t + jX'd I from the solved power flow (angles relative to the infinite bus)."""
    rg = net.res_gen.iloc[0]
    vt = rg.vm_pu * np.exp(1j * np.radians(rg.va_degree))
    i = np.conj((rg.p_mw + 1j * rg.q_mvar) / S_BASE / vt)
    return vt + 1j * GEN["Xdp"] * i, vt, i


def network_ybus(net):
    """pandapower's internal Ybus (out-of-service and isolated buses already removed),
    returned with the bus numbers it keeps, in matrix order."""
    lookup = net._pd2ppc_lookups["bus"]
    ybus = net._ppc["internal"]["Ybus"].toarray()
    # isolated buses are moved to the end of the ppc bus table and cut from the internal matrix
    order = sorted((b for b in net.bus.index if lookup[b] < ybus.shape[0]), key=lambda b: lookup[b])
    idx = [lookup[b] for b in order]
    return ybus[np.ix_(idx, idx)], order


def transfer_admittance(net, faulted_bus=None):
    """Kron-reduce the network to the two sources (E' behind X'd, infinite bus).

    The internal node of the generator is added as an extra row; a bolted fault
    sets its bus voltage to zero, which is the same as deleting that bus.
    """
    ybus, order = network_ybus(net)
    keep = [k for k, b in enumerate(order) if b != faulted_bus]
    ybus = ybus[np.ix_(keep, keep)]
    order = [order[k] for k in keep]
    n = len(order)
    y = np.zeros((n + 1, n + 1), dtype=complex)
    y[:n, :n] = ybus
    g = order.index(GEN["bus"])
    yg = 1 / (1j * GEN["Xdp"])
    y[n, n] += yg
    y[g, g] += yg
    y[n, g] -= yg
    y[g, n] -= yg
    ext = [n, order.index(2)]                       # internal EMF node, infinite bus
    rest = [k for k in range(n + 1) if k not in ext]
    yr = y[np.ix_(ext, ext)] - y[np.ix_(ext, rest)] @ np.linalg.solve(y[np.ix_(rest, rest)], y[np.ix_(rest, ext)])
    return yr[0, 1]


def critical_angle(pm, p_fault, p_post, delta0):
    """Equal-area criterion: A1 (accelerating, fault-on curve) = A2 (decelerating, post-fault curve)."""
    delta_u = math.pi - math.asin(pm / p_post)        # unstable equilibrium on the post-fault curve
    cos_cr = (pm * (delta_u - delta0) + p_post * math.cos(delta_u) - p_fault * math.cos(delta0)) / (p_post - p_fault)
    return math.acos(cos_cr), delta_u


def swing(t_clear, delta0, pm, p_fault, p_post, H=GEN["H"], D=GEN["D"], t_end=5.0, max_step=1e-3):
    """Classical swing equation (2H/w_syn) d2(delta)/dt2 = Pm - Pmax sin(delta) - D dw/w_syn, with
    w_pu = 1 in the inertia term; Pmax switches from the fault-on to the post-fault value at t_clear."""
    def rhs(pmax):
        return lambda t, y: [y[1], OMEGA_SYN / (2 * H) * (pm - pmax * math.sin(y[0]) - D * y[1] / OMEGA_SYN)]
    opts = dict(rtol=1e-10, atol=1e-12, max_step=max_step, dense_output=False)
    a = solve_ivp(rhs(p_fault), (0.0, t_clear), [delta0, 0.0], **opts)
    b = solve_ivp(rhs(p_post), (t_clear, t_end), a.y[:, -1], **opts)
    return np.concatenate([a.t, b.t[1:]]), np.concatenate([a.y[0], b.y[0, 1:]])


def fixed_step(method, t_clear, delta0, pm, p_fault, p_post, H=GEN["H"], dt=1 / 120, t_end=5.0):
    """Same swing equation with a fixed step of dt (0.5 cycle by default) and the event placed exactly
    at t_clear. `method` is "heun" (trapezoidal predictor-corrector) or "midpoint"; both are explicit
    second-order schemes, which add a little energy every swing when there is no damping."""
    def f(y, pmax):
        return np.array([(y[1] - 1.0) * OMEGA_SYN, (pm - pmax * math.sin(y[0])) / (2 * H)])   # y = [delta, w_pu]
    y, t, rows = np.array([delta0, 1.0]), 0.0, [(0.0, delta0, 1.0)]
    while t < t_end - 1e-12:
        h = t_clear - t if t < t_clear < t + dt - 1e-12 else dt
        pmax = p_fault if t < t_clear - 1e-12 else p_post
        k1 = f(y, pmax)
        y = y + h / 2 * (k1 + f(y + h * k1, pmax)) if method == "heun" else y + h * f(y + h / 2 * k1, pmax)
        t += h
        rows.append((t, y[0], y[1]))
    return pd.DataFrame(rows, columns=["t [s]", "delta [rad]", "w [pu]"])


def max_angle_equal_area(pm, pmax, delta1, area1):
    """Angle where the decelerating area under Pmax sin(delta) - Pm, starting at delta1, equals area1."""
    def excess(d):
        return pmax * (math.cos(delta1) - math.cos(d)) - pm * (d - delta1) - area1
    return brentq(excess, delta1, math.pi - math.asin(pm / pmax))


def is_stable(t_clear, delta0, pm, p_fault, p_post, t_end=3.0):
    """Stable when the angle never passes the unstable equilibrium of the post-fault curve."""
    _, d = swing(t_clear, delta0, pm, p_fault, p_post, t_end=t_end)
    return d.max() < math.pi - math.asin(pm / p_post)


def critical_clearing_time(delta0, pm, p_fault, p_post, lo=0.0, hi=1.0, tol=1e-6):
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if is_stable(mid, delta0, pm, p_fault, p_post) else (lo, mid)
    return lo


def modified_euler(t_switch, pmax_fault, pmax_post, delta0, pm=1.0, H=GEN["H"], dt=0.01, steps=86):
    """The textbook's BASIC program (Table 12.1): predictor-corrector, both stages on the same Pmax.

    The program switches to the post-fault curve at the first step that starts with delta at the
    clearing angle. Table 12.1 marks that switch after t = 0.34 s (case 1, delta = 1.954) and after
    t = 0.36 s (case 2, delta = 2.0896, printed 2.090), so the switch is placed by time here.
    """
    d, w, rows = delta0, OMEGA_SYN, []
    for k in range(1, steps + 1):
        pmax = pmax_post if (k - 1) * dt >= t_switch - 1e-9 else pmax_fault
        dd = w - OMEGA_SYN
        dw = (pm - pmax * math.sin(d)) * OMEGA_SYN ** 2 / (2 * H * w)
        d_p, w_p = d + dd * dt, w + dw * dt
        dd_p = w_p - OMEGA_SYN
        dw_p = (pm - pmax * math.sin(d_p)) * OMEGA_SYN ** 2 / (2 * H * w_p)
        d, w = d + (dd + dd_p) * dt / 2, w + (dw + dw_p) * dt / 2
        rows.append((round(k * dt, 10), d, w))
    return pd.DataFrame(rows, columns=["t [s]", "delta [rad]", "omega [rad/s]"]).set_index("t [s]")


def residual_band(build, measure, tol=TOL_MVA, solver=None):
    """Worst case when every non-slack bus keeps a mismatch up to the MVA tolerance
    (P at every non-slack bus, Q at PQ buses; effects added in absolute value)."""
    solver = solver or solve
    base = measure(solver(build()))
    band = pd.Series(0.0, index=base.index)
    probe = build()
    pv = set(probe.gen.bus)
    for bus in probe.bus.index.difference(probe.ext_grid.bus):
        for col in (["p_mw"] if bus in pv else ["p_mw", "q_mvar"]):
            worst = pd.Series(0.0, index=base.index)
            for sign in (1, -1):
                net = build()
                pp.create_load(net, bus, **{"p_mw": 0.0, "q_mvar": 0.0, col: sign * tol})
                worst = np.maximum(worst, (measure(solver(net)) - base).abs())
            band += worst
    return band


def compare(rows):
    """Comparison table; fails loudly when any row is outside its tolerance."""
    df = pd.DataFrame(rows, columns=["Quantity", "pandapower", "Reference", "Tolerance", "Source"])
    df["Abs diff"] = (df.pandapower - df.Reference).abs()
    df["Pass"] = df["Abs diff"] <= df.Tolerance
    assert df.Pass.all(), df.loc[~df.Pass, "Quantity"].tolist()
    return df[["Quantity", "pandapower", "Reference", "Abs diff", "Tolerance", "Pass", "Source"]].round(5)
