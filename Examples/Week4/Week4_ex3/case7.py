"""Seven-bus wind case of Glover, Overbye & Sarma, Problem 6.61 (cases Problem6_61_PQ and Problem6_61_PV).

The wind plant at bus 4 is either a type 1/2 machine (fixed 0.9 power factor, absorbing, with a 66 Mvar fixed
capacitor) or a type 3/4 machine (voltage control at 1.0 pu within +-50 Mvar). Network data and the PowerWorld
results are copied from the case files; the solver handles Mvar limits with PowerWorld's back-off rule.
"""
import math

import numpy as np
import pandas as pd
import pandapower as pp

S_BASE = 100.0
TOL_MVA = 0.1
F_HZ = 60.0
SLACK_BUS = 7
WIND_BUS = 4
WIND_MW = 100.0
WIND_PF = 0.9          # type 1/2: lagging power factor, the machine absorbs Mvar
WIND_QLIM = 50.0       # type 3/4: +-50 Mvar
CAP_MVAR = 66.0        # fixed capacitor at the type 1/2 plant, nominal at 1.0 pu

# bus, name, kV, area
BUSES = [
    (1, 'Bus 1', 138.0, 1),
    (2, 'Bus 2', 138.0, 1),
    (3, 'Bus 3', 138.0, 1),
    (4, 'Bus 4', 138.0, 1),
    (5, 'Bus 5', 138.0, 1),
    (6, 'Bus 6', 138.0, 2),
    (7, 'Bus 7', 138.0, 3),
]
# bus, MW, Mvar (constant power)
LOADS = [
    (2, 40.0, 20.0),
    (3, 110.0, 40.0),
    (4, 80.0, 30.0),
    (5, 130.0, 40.0),
    (6, 200.0, 0.0),
    (7, 200.0, 0.0),
]
# from, to, circuit, R, X, B (pu on 100 MVA), MVA limit
LINES = [
    (1, 2, '1', 0.02, 0.06, 0.06, 150.0),
    (1, 3, '1', 0.08, 0.24, 0.05, 65.0),
    (2, 3, '1', 0.06, 0.18, 0.04, 80.0),
    (2, 4, '1', 0.06, 0.18, 0.04, 100.0),
    (2, 5, '1', 0.04, 0.12, 0.03, 100.0),
    (2, 6, '1', 0.02, 0.06, 0.05, 200.0),
    (3, 4, '1', 0.01, 0.03, 0.02, 100.0),
    (4, 5, '1', 0.08, 0.24, 0.05, 60.0),
    (7, 5, '1', 0.02, 0.06, 0.04, 200.0),
    (6, 7, '1', 0.08, 0.24, 0.05, 200.0),
    (6, 7, '2', 0.08, 0.24, 0.05, 200.0),
]
# conventional units and the wind plant: bus, voltage setpoint, MW min, MW max (order as in PW states below)
GENS = [
    (1, 1.05, 0.0, 400.0),
    (2, 1.04, 100.0, 500.0),
    (4, 1.0, 0.0, 100.0),
    (6, 1.04, 150.0, 500.0),
    (7, 1.04, 0.0, 600.0),
]

# PowerWorld solutions: bus order as BUSES, generator order as GENS. "out24" opens line 2-4; PowerWorld's
# area interchange control (economic dispatch per area) redispatched the units after the outage
PW = {'pq': {'vm': [1.04999998, 1.03999997, 0.99238124, 0.99957035, 1.00658566, 1.03999997, 1.03999996],
        'va': [6.1398549, 4.2022167, 1.1377968, 1.6570343, -0.805556, 2.7980413, 0.0],
        'gen_p': [102.98325, 164.16724, 100.0, 200.32704, 200.227],
        'gen_q': [4.98045, 35.49638, -48.43221, -6.60513, 51.51826]},
 'pv': {'vm': [1.04999997, 1.03999997, 0.99271973, 1.00000001, 1.00665443, 1.03999997, 1.03999996],
        'va': [6.121589, 4.1824659, 1.1156542, 1.6331874, -0.8142166, 2.7789221, 0.0],
        'gen_p': [103.11178, 163.93414, 100.0, 200.07002, 200.69721],
        'gen_q': [4.82462, 35.07755, 18.19547, -6.53457, 51.23143]},
 'pq_out24': {'vm': [1.04999998, 1.03999997, 0.97080579, 0.97263292, 1.00250517, 1.03999997, 1.03999996],
              'va': [6.2723225, 4.5262004, 0.3524358, 0.6030155, -0.8726823, 3.0134564, 0.0],
              'gen_p': [105.82358, 161.77181, 100.0, 200.37735, 201.10571],
              'gen_q': [13.88937, 31.61509, -48.43221, -6.36317, 58.4991]},
 'pv_out24': {'vm': [1.04999998, 1.03999998, 0.99231586, 1.00000001, 1.00653733, 1.03999997, 1.03999996],
              'va': [6.2279941, 4.4863655, 0.0135721, 0.1307455, -0.9350977, 2.9870582, 0.0],
              'gen_p': [106.01233, 161.03486, 100.0, 200.3746, 201.19104],
              'gen_q': [4.75919, 16.26284, 42.80278, -6.3951, 51.49063]}}


def build_case(model, outage=None, dispatch=None):
    """model "PQ" = type 1/2 plant with capacitor, "PV" = type 3/4 plant. dispatch: PowerWorld state key whose
    generator MW to use (the case files' own dispatch by default)."""
    net = pp.create_empty_network(sn_mva=S_BASE, f_hz=F_HZ)
    for num, name, kv, area in BUSES:
        pp.create_bus(net, vn_kv=kv, name=name, index=num, zone=area)
    p = PW[dispatch or model.lower()]["gen_p"]
    for k, (bus, vset, pmin, pmax) in enumerate(GENS):
        if bus == SLACK_BUS:
            pp.create_ext_grid(net, bus, vm_pu=vset, va_degree=0.0, name=f"G{bus}")
        elif bus == WIND_BUS and model == "PQ":
            q = -p[k] * math.tan(math.acos(WIND_PF))
            pp.create_sgen(net, bus, p_mw=p[k], q_mvar=q, name="Wind")
        elif bus == WIND_BUS:
            pp.create_gen(net, bus, p_mw=p[k], vm_pu=vset, min_q_mvar=-WIND_QLIM, max_q_mvar=WIND_QLIM, name="Wind")
        else:
            # the conventional units have effectively unlimited Mvar in the case (+-9900)
            pp.create_gen(net, bus, p_mw=p[k], vm_pu=vset, name=f"G{bus}")
    for bus, mw, mvar in LOADS:
        pp.create_load(net, bus, p_mw=mw, q_mvar=mvar, name=f"L{bus}")
    z = 138.0 ** 2 / S_BASE
    for f, t, ckt, r, x, b, lim in LINES:
        pp.create_line_from_parameters(
            net, f, t, length_km=1.0, r_ohm_per_km=r * z, x_ohm_per_km=x * z,
            c_nf_per_km=b / z / (2 * math.pi * F_HZ) * 1e9, max_i_ka=lim / (math.sqrt(3) * 138.0),
            name=f"{f}-{t}-{ckt}", in_service=f"{f}-{t}-{ckt}" != outage)
    if model == "PQ":
        # a fixed capacitor is a constant impedance: its Mvar follows V^2
        pp.create_shunt(net, WIND_BUS, q_mvar=-CAP_MVAR, p_mw=0.0, name="Cap")
    net.gen["pv_on"] = net.gen.in_service
    return net


def solve_pw(net, vtol=5e-5, max_outer=30):
    """Mvar limits with PowerWorld's back-off: a unit parked on a limit returns to voltage control once its bus
    crosses the setpoint on the side the limit cannot explain."""
    pv = net.gen.index[net.gen.pv_on.astype(bool)].tolist()
    parked = {}
    for _ in range(max_outer):
        net.sgen.drop(net.sgen.index[net.sgen.name.astype(str).str.startswith("park")], inplace=True)
        net.gen.loc[pv, "in_service"] = True
        for i, side in parked.items():
            net.gen.at[i, "in_service"] = False
            pp.create_sgen(net, net.gen.at[i, "bus"], p_mw=net.gen.at[i, "p_mw"],
                           q_mvar=net.gen.at[i, f"{side}_q_mvar"], name=f"park{i}")
        pp.runpp(net, enforce_q_lims=False)
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


def wind_q(net):
    """Mvar of the wind plant, whichever element carries it (sgen, regulating gen, or gen parked on a limit)."""
    hit = net.sgen.index[net.sgen.bus == WIND_BUS]
    if len(hit):
        return net.res_sgen.q_mvar[hit].sum()
    return net.res_gen.q_mvar[net.gen.index[net.gen.bus == WIND_BUS][0]]


def residual_band(build, measure, tol=TOL_MVA):
    """Worst case when every non-slack bus keeps a mismatch up to the MVA tolerance."""
    net = build()
    base = measure(solve_pw(net))
    band = pd.Series(0.0, index=base.index)
    pv_buses = set(net.gen.bus[net.gen.pv_on.astype(bool)])
    for bus in net.bus.index.difference([SLACK_BUS]):
        for col in (["p_mw"] if bus in pv_buses else ["p_mw", "q_mvar"]):
            worst = pd.Series(0.0, index=base.index)
            for sign in (1, -1):
                extra = pp.create_load(net, bus, **{"p_mw": 0.0, "q_mvar": 0.0, col: sign * tol})
                worst = np.maximum(worst, (measure(solve_pw(net)) - base).abs())
                net.load.drop(extra, inplace=True)
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


def measure(net):
    vals = {f"V {b} [pu]": net.res_bus.vm_pu[b] for b in net.bus.index}
    vals.update({f"delta {b} [deg]": net.res_bus.va_degree[b] for b in net.bus.index})
    vals["Wind Q [Mvar]"] = wind_q(net)
    if len(net.shunt):
        vals["Cap Q [Mvar]"] = -net.res_shunt.q_mvar.sum()
    return pd.Series(vals)


def state_rows(net, key, band, source):
    """Bus voltages and angles against a PowerWorld state; the case files store them in single precision,
    so the printed digit is the float32 spacing (1.2e-7 pu, 2e-6 deg)."""
    ref = PW[key]
    rows = []
    for k, (num, *_ ) in enumerate(BUSES):
        rows.append(("Bus voltage", f"V {num} [pu]", net.res_bus.vm_pu[num], ref["vm"][k], 1.2e-7 + band[f"V {num} [pu]"], source))
        rows.append(("Bus angle", f"delta {num} [deg]", net.res_bus.va_degree[num], ref["va"][k], 2e-6 + band[f"delta {num} [deg]"], source))
    k = [g[0] for g in GENS].index(WIND_BUS)
    rows.append(("Wind plant", "Wind Q [Mvar]", wind_q(net), ref["gen_q"][k], 5e-6 + band["Wind Q [Mvar]"], source))
    return rows
