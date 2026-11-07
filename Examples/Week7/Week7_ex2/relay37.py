"""Relay views of faults in the 37-bus design case (Problem 10.64 data, Week 6 sequence networks).

A fault part-way along a line gets its own node: the line's pi model is taken out of the
sequence admittance matrices and replaced by two shorter pi sections that meet at the fault
point. Pre-fault voltages, loads (constant admittance) and generators behind their sequence
impedances are exactly as in fault37.sequence_zbus, so the classical (superposition) method
gives every bus voltage and branch current during the fault.
"""
import contextlib
import io
import logging
import math
import warnings

import geojson
import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.protection.protection_devices.ocrelay import OCRelay

from case37 import LINES
from fault37 import BRANCH_ZERO, POS, ORDER, A_MAT, sequence_zbus

LINE_Z1 = {(f, t, c): (complex(r, x), b) for f, t, c, r, x, b, _ in LINES}
LINE_Z0 = {(f, t, c): (complex(r0, x0), b0) for f, t, c, r0, x0, b0, xf in BRANCH_ZERO if not xf}


def _pi(y, i, j, z, b, sign=1.0):
    """Add (sign = 1) or remove (sign = -1) a pi section between matrix rows i and j."""
    ys = 1 / z
    y[i, i] += sign * (ys + 0.5j * b)
    y[j, j] += sign * (ys + 0.5j * b)
    y[i, j] -= sign * ys
    y[j, i] -= sign * ys


def fault_on_line(net, line, x, charging=True):
    """Sequence Zbus with an extra node at fraction x of `line` (from-bus end = 0).

    Returns z1, z2, z0 (38 x 38, the fault node last) and the pre-fault voltages of all 38 nodes.
    charging=False drops the faulted line's own shunt susceptance (both sections, all sequences).
    """
    z1, z2, z0, v0 = sequence_zbus(net)
    f, t, _ = line
    i, j, n = POS[f], POS[t], len(ORDER)
    mats = []
    for z, (zl, bl) in ((z1, LINE_Z1[line]), (z2, LINE_Z1[line]), (z0, LINE_Z0[line])):
        y = np.zeros((n + 1, n + 1), complex)
        y[:n, :n] = np.linalg.inv(z)
        _pi(y, i, j, zl, bl, -1.0)
        bs = bl if charging else 0.0
        _pi(y, i, n, x * zl, x * bs)
        _pi(y, n, j, (1 - x) * zl, (1 - x) * bs)
        mats.append(y)
    # pre-fault voltage of the new node: no injection there, so it follows from its own row
    y1 = mats[0]
    vf = -(y1[n, :n] @ v0) / y1[n, n]
    v_all = np.append(v0, vf)
    return [np.linalg.inv(y) for y in mats] + [v_all]


def fault_at_bus(net, bus):
    """Sequence Zbus for a fault on an existing bus (same layout as fault_on_line, no extra node)."""
    z1, z2, z0, v0 = sequence_zbus(net)
    return z1, z2, z0, v0, POS[bus]


def solve_fault(z1, z2, z0, v0, k, kind):
    """Sequence voltages (rows 0, 1, 2) at every node for a bolted fault at node k."""
    if kind == "3ph":
        i1 = v0[k] / z1[k, k]
        return np.vstack([np.zeros_like(v0), v0 - z1[:, k] * i1, np.zeros_like(v0)])
    i0 = v0[k] / (z1[k, k] + z2[k, k] + z0[k, k])                  # phase a to ground
    return np.vstack([-z0[:, k] * i0, v0 - z1[:, k] * i0, -z2[:, k] * i0])


def section_current(v012, a, b, line, frac, charging=True):
    """Sequence currents leaving node a into a pi section of `frac` x the line, towards node b."""
    out = []
    for s, (zl, bl) in ((0, LINE_Z0[line]), (1, LINE_Z1[line]), (2, LINE_Z1[line])):
        bs = bl if charging else 0.0
        out.append((v012[s, a] - v012[s, b]) / (frac * zl) + v012[s, a] * 0.5j * frac * bs)
    return np.array(out)


def relay_view(v012, i012, k0):
    """Phase-a voltage and current seen by the relay, and its apparent impedances [pu].

    z_phase is the positive-sequence element (V1 / I1, what a phase element sees for a three-phase
    fault); z_ground_comp is the ground element Va / (Ia + k0 * 3 I0); z_ground_raw drops k0.
    """
    va = (A_MAT @ v012)[0]
    ia = (A_MAT @ i012)[0]
    return {"z_phase": v012[1] / i012[1], "z_ground_comp": va / (ia + k0 * 3 * i012[0]),
            "z_ground_raw": va / ia, "ia": ia, "i0": i012[0]}


def mho_zone(z, zones):
    """First mho zone (name, reach) that contains z; a mho circle has the reach as its diameter."""
    for name, reach in zones:
        if abs(z - reach / 2) <= abs(reach) / 2:
            return name
    return "none"


def fault_case(net, where, kind, charging=True):
    """Bolted fault at `where` = bus number, or (line, x) with x measured from the line's from bus.

    Returns the sequence voltages of every node, the fault node index, and the fault description.
    """
    if isinstance(where, tuple):
        line, x = where
        z1, z2, z0, v0 = fault_on_line(net, line, x, charging)
        k = len(v0) - 1
    else:
        z1, z2, z0, v0, k = fault_at_bus(net, where)
        line, x = None, None
    return solve_fault(z1, z2, z0, v0, k, kind), k, line, x


def branch_current(v012, bus, line, fault, charging=True):
    """Sequence currents from `bus` into `line`, using the faulted section when the fault sits on it."""
    v, k, f_line, x = fault
    other = line[1] if bus == line[0] else line[0]
    if f_line == line:
        frac = x if bus == line[0] else 1 - x
        return section_current(v012, POS[bus], k, line, frac, charging)
    return section_current(v012, POS[bus], POS[other], line, 1.0)


def add_oc_relays(net_sc, relays, pickup_ka, tms, curve="standard_inverse"):
    """One switch and one IDMT OCRelay per (bus, line) end, with manual pickup and time settings.

    The relay constructor always runs its own calc_sc on a copy of the grid (with a fault bus near
    the line end) to size defaults; with manual settings none of that result is used, so its
    pandas and branch-result warnings are silenced here. The trip times come from res_switch_sc,
    which the caller fills with fault currents from the sequence Zbus.
    """
    # the constructor's fault-bus helper places the new bus between the line ends' coordinates;
    # any layout works because only the electrical data matter
    net_sc.bus["geo"] = [geojson.dumps(geojson.Point((math.cos(k), math.sin(k)))) for k in range(len(net_sc.bus))]
    names = {n: i for i, n in net_sc.line.name.items()}
    for (bus, line) in relays:
        pp.create_switch(net_sc, bus, names["-".join(map(str, line))], et="l", closed=True,
                         name=f"{bus} on {line[0]}-{line[1]}")
    ids = net_sc.switch.index
    pickup = pd.DataFrame({"switch_id": ids, "I_s": pickup_ka})
    times = pd.DataFrame({"switch_id": ids, "tms": tms, "t_grade": 0.0})
    level = logging.getLogger("pandapower").level
    logging.getLogger("pandapower").setLevel(logging.ERROR)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        for s in ids:
            OCRelay(net_sc, s, "IDMT", times, pickup_current_manual=pickup, curve_type=curve)
    logging.getLogger("pandapower").setLevel(level)
    return net_sc
