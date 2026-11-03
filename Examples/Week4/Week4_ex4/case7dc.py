"""Seven-bus DC power flow case of Glover, Overbye & Sarma, Problems 6.59 and 6.60 (case Problem6_59).

Lossless lines (R = 0, B = 0) with only series reactance; bus 7 is the slack. The case is solved in PowerWorld
with the DC approximation (resistance ignored, no loss compensation). Network data and the PowerWorld results are
copied from the case files.
"""
import math

import numpy as np
import pandas as pd
import pandapower as pp

S_BASE = 100.0
F_HZ = 60.0
KV = 138.0
SLACK_BUS = 7
TOL_MW = 5e-6          # generator MW are printed to 1e-5 in the case file

# bus, name, area
BUSES = [
    (1, 'Bus 1', 1),
    (2, 'Bus 2', 1),
    (3, 'Bus 3', 1),
    (4, 'Bus 4', 1),
    (5, 'Bus 5', 1),
    (6, 'Bus 6', 2),
    (7, 'Bus 7', 3),
]
# bus, MW, Mvar (constant power; Mvar is used only by the AC comparison)
LOADS = [
    (2, 40.0, 20.0),
    (3, 110.0, 40.0),
    (4, 80.0, 30.0),
    (5, 130.0, 40.0),
    (6, 200.0, 0.0),
    (7, 200.0, 0.0),
]
# from, to, circuit, X (pu on 100 MVA), rating A [MVA]
LINES = [
    (1, 2, '1', 0.06, 150.0),
    (1, 3, '1', 0.24, 65.0),
    (2, 3, '1', 0.18, 80.0),
    (2, 4, '1', 0.18, 100.0),
    (2, 5, '1', 0.12, 100.0),
    (2, 6, '1', 0.06, 200.0),
    (3, 4, '1', 0.03, 100.0),
    (4, 5, '1', 0.24, 60.0),
    (7, 5, '1', 0.06, 200.0),
    (6, 7, '1', 0.24, 200.0),
    (6, 7, '2', 0.24, 200.0),
]
LINE_NAMES = [f"{f}-{t}-{c}" for f, t, c, _, _ in LINES]
RATING = np.array([lim for *_, lim in LINES])
# bus, voltage setpoint, MW min, MW max, MW (as saved in the case; used unchanged)
GENS = [
    (1, 1.05, 0.0, 400.0, 160.13447),
    (2, 1.04, 100.0, 500.0, 149.86552),
    (4, 1.00, 50.0, 200.0, 50.0),
    (6, 1.04, 150.0, 500.0, 200.01168),
    (7, 1.04, 0.0, 600.0, 199.98833),
]

# PowerWorld DC solution: angles from the aux file (single precision), from-end MW from the MATPOWER export
PW = {'va': [7.6786001, 4.2011501, -0.4316698, -0.3268295, -1.4001159, 2.8010345, 0.0],
      'flow': [101.15, 58.98, 44.92, 43.90, 81.47, 40.73, -6.10, 7.81, 40.73, 20.37, 20.37],
      'slack_p': 199.98833}


def build_case(outage=None, gen_p=None):
    """outage: line name to open. gen_p: {bus: MW} overrides for the non-slack units."""
    net = pp.create_empty_network(sn_mva=S_BASE, f_hz=F_HZ)
    for num, name, area in BUSES:
        pp.create_bus(net, vn_kv=KV, name=name, index=num, zone=area)
    for bus, vset, pmin, pmax, p in GENS:
        if bus == SLACK_BUS:
            pp.create_ext_grid(net, bus, vm_pu=vset, va_degree=0.0, name=f"G{bus}")
        else:
            # the units have effectively unlimited Mvar in the case (+-9900)
            pp.create_gen(net, bus, p_mw=(gen_p or {}).get(bus, p), vm_pu=vset, name=f"G{bus}")
    for bus, mw, mvar in LOADS:
        pp.create_load(net, bus, p_mw=mw, q_mvar=mvar, name=f"L{bus}")
    z = KV ** 2 / S_BASE
    for (f, t, ckt, x, lim), name in zip(LINES, LINE_NAMES):
        pp.create_line_from_parameters(
            net, f, t, length_km=1.0, r_ohm_per_km=0.0, x_ohm_per_km=x * z, c_nf_per_km=0.0,
            max_i_ka=lim / (math.sqrt(3) * KV), name=name, in_service=name != outage)
    return net


def dc_flows(net):
    pp.rundcpp(net)
    return net.res_line.p_from_mw.to_numpy()


def dc_loading(flows):
    """MW over rating A. pandapower's DC loading_percent converts MW to current with the generator setpoint
    voltages, so it is not the MW loading PowerWorld's DC solution reports."""
    return np.abs(flows) / RATING * 100


def measure(net):
    pp.rundcpp(net)
    vals = {f"delta {b} [deg]": net.res_bus.va_degree[b] for b in net.bus.index}
    vals.update({f"P {n} [MW]": p for n, p in zip(LINE_NAMES, net.res_line.p_from_mw)})
    vals["Slack P [MW]"] = net.res_ext_grid.p_mw.iloc[0]
    return pd.Series(vals)


def residual_band(build, measure, tol=TOL_MW):
    """Worst case when every non-slack unit is off by half its printed digit. The DC solution is linear, so this
    is the only freedom between the case file and a re-solve."""
    net = build()
    base = measure(net)
    band = pd.Series(0.0, index=base.index)
    for i in net.gen.index:
        p0 = net.gen.at[i, "p_mw"]
        worst = pd.Series(0.0, index=base.index)
        for sign in (1, -1):
            net.gen.at[i, "p_mw"] = p0 + sign * tol
            worst = np.maximum(worst, (measure(net) - base).abs())
        net.gen.at[i, "p_mw"] = p0
        band += worst
    return band


def compare(rows):
    """Tolerance = half of the printed digit + residual band; fails loudly when a row is outside it."""
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
        "Smallest margin": (df.Tolerance - df["Abs diff"]).groupby([df.Group, df.Source], sort=False).min(),
        "All pass": g.Pass.all(),
    }).reset_index()


def state_rows(net, band, source):
    """Angles carry the float32 spacing (2e-6 deg); MATPOWER flows are printed to 0.01 MW."""
    vals = measure(net)
    rows = []
    for k, (num, *_) in enumerate(BUSES):
        q = f"delta {num} [deg]"
        rows.append(("Bus angle", q, vals[q], PW["va"][k], 2e-6 + band[q], source))
    for k, name in enumerate(LINE_NAMES):
        q = f"P {name} [MW]"
        rows.append(("Line flow", q, vals[q], PW["flow"][k], 0.005 + band[q], source))
    rows.append(("Slack unit", "Slack P [MW]", vals["Slack P [MW]"], PW["slack_p"], TOL_MW + band["Slack P [MW]"], source))
    return rows
