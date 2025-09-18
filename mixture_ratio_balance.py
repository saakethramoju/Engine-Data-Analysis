import numpy as np
from rocketcea.cea_obj_w_units import CEA_Obj
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.integrate import cumulative_trapezoid


throat_area         = 0.003903218   # m^2
ambient_pressure    = 14.67         # psia
expansion_ratio     = 4.73
start_time          = 187           # s
end_time            = 207           # s
guess_mr_bracket    = [1, 3]
start_fuel_volume   = 56            # L
fuel_density        = 800           # kg/m^3        
csv_filename        = "2025-09-13-vespula-abl-hotfire.csv"

data = pd.read_csv(csv_filename) 
time = data["timestamp"].to_numpy() - data["timestamp"].iloc[0]
idx1 = np.argmin(np.abs(time - start_time))
#idx2 = np.argmin(np.abs(time - 189))
idx2 = np.argmin(np.abs(time - end_time))
mask = (np.arange(len(time)) > idx1) & (np.arange(len(time)) < idx2)
time = time[mask]
time = time - time[0]

chpt   = (data["CHPT1(psi)"]   + 14.67)
thrust = (data["Thrust(lbf)"]) * -1
starting_volume = start_fuel_volume / 1000
starting_mass = starting_volume * fuel_density

chpt    = chpt[mask].to_numpy()
thrust  = thrust[mask].to_numpy()

rocket = CEA_Obj(oxName='LOX', fuelName='RP-1', temperature_units='degK', 
                cstar_units='m/sec', specific_heat_units='kJ/kg degK', 
                sonic_velocity_units='m/s', enthalpy_units='J/kg', 
                density_units='kg/m^3')

def get_mdot(Pc, mr, At, cea_obj: CEA_Obj = rocket):
    Pt = Pc / cea_obj.get_Throat_PcOvPe(Pc, mr)
    _, Tt, _ = cea_obj.get_Temperatures(Pc, mr)
    mw, gam = cea_obj.get_Throat_MolWt_gamma(Pc, mr)
    mdot = (Pt * 6894.76) * At * np.sqrt(mw * gam / (Tt * 8314.46261815324))
    return mdot

def get_Ve(Pc, mr, eps, cea_obj: CEA_Obj = rocket):
    _, _, ae = cea_obj.get_SonicVelocities(Pc, mr, eps, 1, 1)
    Me = cea_obj.get_MachNumber(Pc, mr, eps, 1, 1)
    Ve = ae * Me
    return Ve

def get_Pe(Pc, mr, eps, cea_obj: CEA_Obj = rocket):
    Pe = Pc / cea_obj.get_PcOvPe(Pc, mr, eps, 1, 1)
    return Pe

def get_thrust(Pc, mr, At, eps, Pamb, cstar_eff = 1, cea_obj: CEA_Obj = rocket):
    mdot = get_mdot(Pc, mr, At, cea_obj)
    Ve = get_Ve(Pc, mr, eps, cea_obj)
    Pe = get_Pe(Pc, mr, eps, cea_obj)
    thrust = cstar_eff*mdot*Ve + At*eps*(Pe - Pamb)*6894.76
    thrust = thrust / 4.44822
    return thrust


def solve_mr(measured_thrust, Pc, At, eps, Pamb, cea_obj: CEA_Obj = rocket):
    def residual(x):
        return get_thrust(Pc, x, At, eps, Pamb, cea_obj) - measured_thrust

    a, b = guess_mr_bracket
    fa, fb = residual(a), residual(b)

    if fa * fb > 0:  
        for new_bracket in [(0.5, 4), (0.5, 6), (1, 5)]:
            a, b = new_bracket
            fa, fb = residual(a), residual(b)
            if fa * fb < 0:
                break
        else:
            return np.nan 
    
    sol = root_scalar(residual, bracket=[a, b], method='brentq')
    return sol.root if sol.converged else np.nan

Pc = chpt[100]   # pick some chamber pressure point
mrs = np.linspace(0.5, 6, 200)
curve = [get_thrust(Pc, m, throat_area, expansion_ratio, ambient_pressure, cstar_eff=0.97) for m in mrs]

plt.plot(mrs, curve, label="Model thrust")
plt.axhline(thrust[100], color='r', label="Measured thrust")
plt.legend()
plt.show()


'''
mr_list     = np.zeros(len(time))
fuel_mdot   = np.zeros(len(time))
for i, t in enumerate(time):
    mr_list[i] = solve_mr(thrust[i], chpt[i], throat_area, expansion_ratio, ambient_pressure)
    total_mdot = get_mdot(chpt[i], mr_list[i], throat_area)
    fuel_mdot[i] = total_mdot / (1 + mr_list[i])

fuel_mdot_fix = np.nan_to_num(fuel_mdot, nan=0.0)
consumed_mass = cumulative_trapezoid(fuel_mdot_fix, time, initial=0)
remaining_mass   = np.maximum(starting_mass - consumed_mass, 0.0)
remaining_volume_L = remaining_mass / fuel_density * 1000  # liters



# -------------- MR plot ---------------
mr_window = 50  
mr_smooth = pd.Series(mr_list).rolling(mr_window, min_periods=1, center=True).mean()
mean_mr   = np.nanmean(mr_list)

fig, ax = plt.subplots(figsize=(10, 6))
# Raw data
ax.plot(time, mr_list, linewidth=1, alpha=0.4, color="tab:green", label="Raw")
# Smoothed data
ax.plot(time, mr_smooth, linewidth=2, color="tab:green",
        label=f"Rolling Avg (Window Size: {mr_window})")
# Mean line
ax.axhline(mean_mr, color="k", linestyle="--", linewidth=1.2,
           label=f"Mean = {mean_mr:.3f}")
# Labels and styling
ax.set_ylabel("Mixture Ratio", fontsize=14)
ax.set_xlabel("Time (s)", fontsize=14)
ax.set_title("Mixture Ratio vs Time", fontsize=16)
ax.grid(True, alpha=0.4)
ax.legend()
plt.tight_layout()



# ------------------ Fuel Mdot -------------
# Rolling average and mean for fuel mdot
fuel_window = 50  
fuel_mdot_smooth = pd.Series(fuel_mdot).rolling(fuel_window, min_periods=1, center=True).mean()
mean_fuel_mdot   = np.nanmean(fuel_mdot)

# Plot fuel mass flow
fig, ax = plt.subplots(figsize=(10, 6))
# Raw data
ax.plot(time, fuel_mdot, linewidth=1, alpha=0.4, color="tab:blue", label="Raw")
# Smoothed data
ax.plot(time, fuel_mdot_smooth, linewidth=2, color="tab:blue",
        label=f"Rolling Avg (Window Size: {fuel_window})")
# Mean line
ax.axhline(mean_fuel_mdot, color="k", linestyle="--", linewidth=1.2,
           label=f"Mean = {mean_fuel_mdot:.3f} kg/s")
# Labels and styling
ax.set_ylabel("ṁ_fuel (kg/s)", fontsize=14)
ax.set_xlabel("Time (s)", fontsize=14)
ax.set_title("Fuel Mass Flow vs Time", fontsize=16)
ax.grid(True, alpha=0.4)
ax.legend()
plt.tight_layout()



# ------------- Fuel Volume Plots -------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time, remaining_volume_L, linewidth=2, color="tab:red", label="Fuel Volume")
ax.axhline(start_fuel_volume, color="k", linestyle="--", linewidth=1.2,
           label=f"Start = {start_fuel_volume:.1f} L")

start_empty_time, empty_time = None, None  # placeholders for duration calc

# --- Start of emptying ---
start_tol = 0.01 * start_fuel_volume  # 1% drop from start
start_indices = np.where(remaining_volume_L < (start_fuel_volume - start_tol))[0]

if len(start_indices) > 0:
    start_empty_idx = start_indices[0]
    start_empty_time = time[start_empty_idx]
    ax.axvline(start_empty_time, color="tab:blue", linestyle="--", linewidth=1.2,
               label=f"Start emptying at {start_empty_time:.2f} s")
    ax.annotate("Start emptying",
                xy=(start_empty_time, remaining_volume_L[start_empty_idx]),
                xytext=(start_empty_time+2, start_fuel_volume*0.95),
                arrowprops=dict(arrowstyle="->", color="tab:blue"),
                fontsize=12, color="tab:blue")

# --- Near empty ---
threshold = 0.5  # L
empty_indices = np.where(remaining_volume_L <= threshold)[0]

if len(empty_indices) > 0:
    empty_idx = empty_indices[0]
    empty_time = time[empty_idx]
    ax.axvline(empty_time, color="tab:purple", linestyle="--", linewidth=1.2,
               label=f"≤ {threshold} L at {empty_time:.2f} s")
    ax.annotate("Tank nearly empty",
                xy=(empty_time, remaining_volume_L[empty_idx]),
                xytext=(empty_time-8, start_fuel_volume*0.3),
                arrowprops=dict(arrowstyle="->", color="tab:purple"),
                fontsize=12, color="tab:purple")

# --- Duration between start and near empty ---
if start_empty_time is not None and empty_time is not None:
    duration = empty_time - start_empty_time
    ax.plot([], [], ' ', label=f"Duration: {duration:.2f} s")

ax.set_ylabel("Fuel Volume (L)", fontsize=14)
ax.set_xlabel("Time (s)", fontsize=14)
ax.set_title("Fuel Volume vs Time", fontsize=16)
ax.grid(True, alpha=0.4)
ax.legend()
plt.tight_layout()
'''


plt.show()
