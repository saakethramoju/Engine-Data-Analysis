# ============================================================
# Imports
# ============================================================
import numpy as np
from rocketcea.cea_obj_w_units import CEA_Obj
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, brentq
from scipy.optimize import minimize_scalar


# ============================================================
# Inputs / Configuration
# ============================================================
throat_area         = 6.05  # in^2
#throat_area         = ((np.pi)*3**2 / 4)
cstar_eff           = 1
ambient_pressure    = 14.67         # psia
expansion_ratio     = 4.73
#expansion_ratio     = (6.05*4.73/(throat_area))
contraction_ratio   = 4.3
fac                 = True          # True if finite area combustors
frozen              = False
frozen_from_throat  = False

start_time          = 112           # s
end_time            = 129           # s     

# csv_filename        = "Test Data Parsing/25_9_21_regen_fire.csv"
# csv_filename        = "Test Data Parsing/2025-09-13-vespula-abl-hotfire.csv"
csv_filename        = "Test Data Parsing/2025-09-27-Vespula-Regen-Fire.csv"
chpt_sensor_name    = "CHPT1(psi)"
thrust_sensor_name  = "Thrust(lbf)"


# ============================================================
# Data Load & Windowing
# ============================================================
data = pd.read_csv(csv_filename) 
time = data["timestamp"].to_numpy() - data["timestamp"].iloc[0]
idx1 = np.argmin(np.abs(time - start_time))
idx2 = np.argmin(np.abs(time - end_time))
#idx1 = np.argmin(np.abs(time - 0))
#idx2 = np.argmin(np.abs(time - time[-1]))
mask = (np.arange(len(time)) > idx1) & (np.arange(len(time)) < idx2)
time = time[mask]
time = time - time[0]

chpt   = (data[chpt_sensor_name]   + 14.67)
thrust = np.abs((data[thrust_sensor_name]))
chpt    = chpt[mask].to_numpy()
thrust  = thrust[mask].to_numpy()

if fac:
    rocket = CEA_Obj(oxName='LOX', fuelName='RP-1',
                     temperature_units='degK', cstar_units='m/sec',
                     specific_heat_units='kJ/kg degK',
                     sonic_velocity_units='m/s', enthalpy_units='J/kg',
                     density_units='kg/m^3', fac_CR=contraction_ratio)
else:
    rocket = CEA_Obj(oxName='LOX', fuelName='RP-1',
                     temperature_units='degK', cstar_units='m/sec',
                     specific_heat_units='kJ/kg degK',
                     sonic_velocity_units='m/s', enthalpy_units='J/kg',
                     density_units='kg/m^3')


# ============================================================
# Function Library
# ============================================================
def get_mdot(Pc, mr, At, cstar_eff, cea_obj: CEA_Obj = rocket):
    """
    Compute total mass flow rate (kg/s) from chamber pressure, MR, and throat area.

    Parameters
    ----------
    Pc : float
        Chamber pressure [psia].
    mr : float
        Mixture ratio (O/F).
    At : float
        Throat area [in^2].
    cstar_eff : float
        c* efficiency multiplier (dimensionless, typically 0.8–1.0).
    cea_obj : CEA_Obj, optional
        Rocket CEA object (defaults to the global 'rocket').

    Returns
    -------
    mdot : float
        Total mass flow rate [kg/s].

    Notes
    -----
    Uses c*_ideal from CEA, scales by cstar_eff, and converts lbf to N via 4.44822.
    """
    cstar_ideal = cea_obj.get_Cstar(Pc, mr)
    cstar = cstar_eff * cstar_ideal
    mdot = Pc * At * 4.44822 / cstar
    return mdot


def get_thrust(Pc, mr, At, Pamb, eps, cf_eff=1, frozen=False, frozen_from_throat=False, cea_obj: CEA_Obj = rocket):
    """
    Compute thrust in lbf given chamber pressure, mixture ratio, geometry, and environment.

    Parameters
    ----------
    Pc : float
        Chamber pressure [psia].
    mr : float
        Mixture ratio (O/F).
    At : float
        Throat area [in^2].
    Pamb : float
        Ambient pressure [psia].
    eps : float
        Expansion ratio (Ae/At).
    cf_eff : float, optional
        Thrust coefficient efficiency multiplier (dimensionless).
    frozen : bool, optional
        If True, use frozen chemistry; else equilibrium.
    frozen_from_throat : bool, optional
        If True, freeze chemistry starting from the throat (mode=1).
    cea_obj : CEA_Obj, optional
        Rocket CEA object (defaults to the global 'rocket').

    Returns
    -------
    thrust : float
        Thrust [lbf].
    """
    if frozen:
        if frozen_from_throat:
            _, cf, _ = cea_obj.getFrozen_PambCf(Pamb, Pc, mr, eps, 1)
        else:
            _, cf, _ = cea_obj.getFrozen_PambCf(Pamb, Pc, mr, eps, 0)
    else:
        _, cf, _ = cea_obj.get_PambCf(Pamb, Pc, mr, eps)

    thrust = cf_eff * cf * Pc * At
    return thrust


def plot_thrust_vs_mr(
    Pc, At, Pamb, eps,
    cf_list=(0.80, 0.85, 0.90, 0.95, 1.00),
    mr_min=0.8, mr_max=2.8, npts=1000,
    frozen=frozen, frozen_from_throat=frozen_from_throat, cea_obj=rocket,
    highlight_mr=None
):
    """
    Plot thrust (lbf) vs mixture ratio for a fixed Pc, At, Pamb, and epsilon.

    Parameters
    ----------
    Pc : float
        Chamber pressure [psia].
    At : float
        Throat area [in^2].
    Pamb : float
        Ambient pressure [psia].
    eps : float
        Expansion ratio (Ae/At).
    cf_list : tuple of float
        Cf efficiency multipliers to sweep and plot.
    mr_min, mr_max : float
        MR sweep bounds.
    npts : int
        Number of MR samples.
    frozen, frozen_from_throat : bool
        Chemistry flags passed to CEA.
    cea_obj : CEA_Obj
        Rocket CEA object.
    highlight_mr : float or None
        Optional MR to annotate on the plot.

    Returns
    -------
    None
    """
    mrs = np.linspace(mr_min, mr_max, npts)

    plt.figure(figsize=(8,5))
    for cf_eff in cf_list:
        T = [get_thrust(Pc, mr, At, Pamb, eps, cf_eff, frozen, frozen_from_throat, cea_obj)
             for mr in mrs]
        plt.plot(mrs, T, linewidth=2, label=f"cf_eff = {cf_eff*100:.0f}%")

    if highlight_mr is not None:
        # mark a reference MR (e.g., your current mr)
        T_ref = get_thrust(Pc, highlight_mr, At, Pamb, eps, cf_list[-2], frozen, frozen_from_throat, cea_obj)
        plt.axvline(highlight_mr, linestyle="--", linewidth=1)
        plt.text(highlight_mr, T_ref, f"  MR={highlight_mr:.2f}", va="bottom")

    plt.xlabel("Mixture Ratio (O/F)")
    plt.ylabel("Thrust (lbf)")
    plt.title(f"Thrust vs MR @ Pc={Pc:.1f} psia, At={At:.3f} in², ε={eps:.3f}, Pamb={Pamb:.2f} psia")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_cfeff_from_pc_mr_and_thrust(Pc, mr, thrust, At, Pamb, eps, 
                                    frozen=False, frozen_from_throat=False, 
                                    cea_obj: CEA_Obj = rocket):
    """
    Solve for Cf efficiency (cf_eff) given Pc, MR, thrust, At, Pamb, and epsilon.

    Parameters
    ----------
    Pc : float
        Chamber pressure [psia].
    mr : float
        Mixture ratio (O/F).
    thrust : float
        Target/measured thrust [lbf].
    At : float
        Throat area [in^2].
    Pamb : float
        Ambient pressure [psia].
    eps : float
        Expansion ratio (Ae/At).
    frozen, frozen_from_throat : bool
        Chemistry flags passed to CEA.
    cea_obj : CEA_Obj
        Rocket CEA object.

    Returns
    -------
    cf_eff : float or np.nan
        Cf efficiency required to match the measured thrust. np.nan if no bracketed root.

    Notes
    -----
    Uses a scalar root solve (bisection) over cf_eff ∈ [0.01, 2.0].
    """
    def f(cf_eff):
        return get_thrust(Pc, mr, At, Pamb, eps, cf_eff, frozen, frozen_from_throat, cea_obj) - thrust

    try:
        sol = root_scalar(f, bracket=[0.01, 2.0], method='bisect')
        if sol.converged:
            return sol.root
        else:
            return np.nan
    except ValueError:
        return np.nan


def cf_ideal(Pc, mr, Pamb, eps, cea_obj, frozen=False, frozen_from_throat=False):
    """
    Return the ideal thrust coefficient Cf* from CEA for the given state and MR.

    Parameters
    ----------
    Pc : float
        Chamber pressure [psia].
    mr : float
        Mixture ratio (O/F).
    Pamb : float
        Ambient pressure [psia].
    eps : float
        Expansion ratio (Ae/At).
    cea_obj : CEA_Obj
        Rocket CEA object.
    frozen, frozen_from_throat : bool
        Chemistry flags passed to CEA.

    Returns
    -------
    cf : float
        Ideal thrust coefficient (dimensionless).
    """
    if frozen:
        mode = 1 if frozen_from_throat else 0
        _, cf, _ = cea_obj.getFrozen_PambCf(Pamb, Pc, mr, eps, mode)
    else:
        _, cf, _ = cea_obj.get_PambCf(Pamb, Pc, mr, eps)
    return cf

'''
def fit_constant_mr_from_window(Pc_series, T_series, At, Pamb, eps, cea_obj,
                                mr_bounds=(1.3, 2.8),
                                eta_bounds=(0.85, 1.00),
                                weights=None,
                                frozen=False, frozen_from_throat=False):
    """
    Fit a constant MR over a time window using many (Pc_i, T_i) pairs.
    For each trial MR, solve the LS-optimal eta_Cf in closed form, clamp to bounds,
    and minimize weighted residuals over MR (1-D).

    Parameters
    ----------
    Pc_series : array_like
        Chamber pressures [psia] across the window.
    T_series : array_like
        Thrust measurements [lbf] across the window.
    At : float
        Throat area [in^2].
    Pamb : float
        Ambient pressure [psia].
    eps : float
        Expansion ratio (Ae/At).
    cea_obj : CEA_Obj
        Rocket CEA object.
    mr_bounds : (float, float)
        Search bounds for MR.
    eta_bounds : (float, float)
        Physical bounds for eta_Cf (e.g., 0.85–1.00).
    weights : array_like or None
        Optional per-sample weights (e.g., 1/sigma_T^2). If None, all 1.
    frozen, frozen_from_throat : bool
        Chemistry flags passed to CEA.

    Returns
    -------
    result : dict
        {
          "mr_hat": float,
          "eta_cf_hat": float,
          "rss": float,
          "r2_unweighted": float,
          "success": bool,
          "message": str
        }

    Notes
    -----
    - x_i = Cf*(MR; Pc_i, eps, Pamb) * Pc_i * At
    - eta_hat(MR) = (Σ w_i x_i T_i) / (Σ w_i x_i^2)
    - Residuals computed with eta_hat clamped to eta_bounds.
    """
    Pc_series = np.asarray(Pc_series, float)
    T_series  = np.asarray(T_series,  float)
    n = len(Pc_series)
    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, float)

    def ls_cost(mr):
        # build x_i = Cf*(mr; Pc_i)*Pc_i*At
        Cf_list = [cf_ideal(Pc_i, mr, Pamb, eps, cea_obj, frozen, frozen_from_throat) for Pc_i in Pc_series]
        x = np.array(Cf_list) * Pc_series * At

        # closed-form eta_hat (weighted)
        num = np.sum(w * x * T_series)
        den = np.sum(w * x * x)
        if den <= 0:
            return np.inf, np.nan, np.nan  # pathological

        eta_hat = num / den
        # clamp eta to physical band and recompute residual if clipped
        eta_clamped = np.clip(eta_hat, eta_bounds[0], eta_bounds[1])

        # residuals with clamped eta
        r = np.sqrt(w) * (T_series - eta_clamped * x)
        rss = np.dot(r, r)

        # mild penalty if clamping was needed (keeps MR that demands eta>1 away)
        pen = 0.0
        if eta_hat < eta_bounds[0]:
            pen = ((eta_bounds[0] - eta_hat) / 0.01)**2
        elif eta_hat > eta_bounds[1]:
            pen = ((eta_hat - eta_bounds[1]) / 0.01)**2

        return rss + pen, eta_clamped, den

    # 1-D bounded search over MR
    res = minimize_scalar(lambda mr: ls_cost(mr)[0],
                          bounds=mr_bounds, method="bounded", options={"xatol":1e-3})

    mr_hat = res.x
    cost, eta_hat, den = ls_cost(mr_hat)

    # R^2 (unweighted) for a quick sense of fit quality
    Cf_list = [cf_ideal(Pc_i, mr_hat, Pamb, eps, cea_obj, frozen, frozen_from_throat) for Pc_i in Pc_series]
    x = np.array(Cf_list) * Pc_series * At
    T_pred = eta_hat * x
    ss_res = np.sum((T_series - T_pred)**2)
    ss_tot = np.sum((T_series - np.mean(T_series))**2) + 1e-12
    r2 = 1 - ss_res/ss_tot

    return {
        "mr_hat": float(mr_hat),
        "eta_cf_hat": float(eta_hat),
        "rss": float(cost),
        "r2_unweighted": float(r2),
        "success": bool(res.success),
        "message": res.message
    }
'''

# ============================================================
# Script Entry
# ============================================================
if __name__ == "__main__":
    # Example single-point inversion helpers
    t = 2200
    p = 227
    mr = 1.9

    print(get_cfeff_from_pc_mr_and_thrust(p, mr, t, throat_area, ambient_pressure, expansion_ratio, frozen, frozen_from_throat))
    print(get_thrust(p, mr, throat_area, ambient_pressure, expansion_ratio, 1, True, True))

    ''' 
    Don't try to back out MR using thrust and chamber pressure. For one, for a given chamber pressure, 
    the thrust produced is highly dependent on eta_Cf. Even if eta_Cf was close to the actual value, 
    the slope of the thrust vs MR curve is very low. Because of this, even a slight change in the inputted 
    measured thrust can drastically change the MR that is outputted, for a given eta_Cf. 
    '''


    '''# Windowed constant-MR fit
    fit = fit_constant_mr_from_window(
        Pc_series=chpt,              # absolute psia over your steady window
        T_series=thrust,             # lbf (same window)
        At=throat_area,
        Pamb=ambient_pressure,
        eps=expansion_ratio,
        cea_obj=rocket,
        mr_bounds=(1.3, 3),        # tighten if you already know a ballpark
        eta_bounds=(0.7, 0.98),     # realistic band
        weights=None,                # or 1/sigma_T^2 per sample if you have it
        frozen=frozen, frozen_from_throat=frozen_from_throat
    )
    print(fit)'''
