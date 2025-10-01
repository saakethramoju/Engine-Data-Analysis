%% Data Parsing and Setup
clc;
close all;

%-------------------------
% File selection
%-------------------------
% csv = '2025-09-13-vespula-abl-hotfire.csv';
% csv = '20250412HETS_Fire_2.csv';
% csv = '25_9_21_regen_fire.csv';
 csv = '2025-09-27-Vespula-Regen-Fire.csv';

%-------------------------
% Import data
%-------------------------
data = readtable(csv, 'VariableNamingRule', 'preserve');

%-------------------------
% Time window (s)
%-------------------------
% times = [180 225];   % ablative 1
 times = [112 129];
% times = [91 99];     % regen blip
% times = [109 150];      % regen long duration 

%-------------------------
% Parameters
%-------------------------
window                 = 50;       % smoothing window (1 = none)
% chpt_name              = "CHPT1(psi)";  % sometimes it changes
chpt_name              = "CHPT1(psi)";  % sometimes it changes
inj_cda_fuel           = 0.5550;   % injector CdA [cm^2 or consistent units]
inj_cda_ox             = 1.250;
rho_fuel               = 800;      % [kg/m^3]
rho_ox                 = 1141;     % [kg/m^3]
lox_venturi_throat_id  = 0.4375;   % in
lox_venturi_inlet_id   = 0.844;    % in
lox_venturi_cd         = 0.978;
flowmeter_multiplier   = 1;
lox_tank_volume        = 2 * 70;   % L
fuel_tank_volume       = 70;       % L
ox_bb_orifice_area     = 0.021 / 1549.997; % m^2
ox_bb_orifice_Cd       = 0.7;
otpt_target            = 375;      % psi
rho_gn2                = 29.323;   % kg/m^3

% Time Processing
time = data.timestamp - data.timestamp(1);

[~, idx1] = min(abs(time - times(1)));
[~, idx2] = min(abs(time - times(2)));

mask = ( (1:numel(time)) > idx1 ) & ( (1:numel(time)) < idx2 );

time = time(mask);
time = time - time(1);
dt   = [0; diff(time)];

% Sensor Data Extraction
fipt    = data.("FIPT(psi)")   + 14.67;  fipt    = fipt(mask);
oipt    = data.("OIPT(psi)")   + 14.67;  oipt    = oipt(mask);
chpt    = data.(chpt_name)  + 14.67;  chpt    = chpt(mask);
thrust  = abs(data.("Thrust(lbf)"));     thrust  = thrust(mask);
ftpt    = data.("FTPT(psi)")   + 14.67;  ftpt    = ftpt(mask);
frunpt  = abs(data.("FRUNPT(psi)") + 14.67); frunpt = frunpt(mask);
% For regen only:
% frunpt = abs(data.("FRUNPT2(psi)") + 14.67); frunpt = frunpt(mask);

odp     = data.("ODP(psi)");              odp     = odp(mask);
otpt    = abs(data.("OTPT(psi)")) + 14.67; otpt   = otpt(mask);
prpto   = abs(data.("PRPTO(psi)")) + 14.67; prpto = prpto(mask);
prpt2   = abs(data.("PRPT2(psi)")) + 14.67; prpt2 = prpt2(mask);

%-------------------------
% OBANG (valve state)
%-------------------------
obang = data.("OBANG");
obang = double(strcmpi(strtrim(obang), 'open'));
obang = obang(mask);

%-------------------------
% Flowmeter (optional)
%-------------------------
if ismember("FLOW(psi)", data.Properties.VariableNames)
    flowmeter = data.("FLOW(psi)");
    flowmeter = flowmeter(mask);
else
    warning('FLOW(psi) not found in table');
    flowmeter = [];
end


%% Chamber Pressure
figure;
plot(time, chpt, 'LineWidth', 2);
xlabel('Time (s)',  'FontSize', 15);
ylabel('Chamber Pressure (psia)', 'FontSize', 15);
title('P_c vs Time', 'FontSize', 15);
grid on;

%% Injector Fuel Manifold Pressure
figure;
plot(time, fipt, 'LineWidth', 1.5);
xlabel('Time (s)',  'FontSize', 15);
ylabel('Pressure (psia)', 'FontSize', 15);
title('Injector Fuel Manifold Pressure vs Time', 'FontSize', 15);
grid on;

%% Thrust
figure;
plot(time, thrust, 'LineWidth', 1.5);
xlabel('Time (s)',  'FontSize', 15);
ylabel('Thrust (lbf)', 'FontSize', 15);
title('Thrust vs Time', 'FontSize', 15);
grid on;

%% Injector Mass Flows and Mixture Ratio
% Mass flows through injector CdA
fuel_mdot = sign(fipt - chpt) .* inj_cda_fuel .* 1e-4 .* ...
            sqrt(2 .* rho_fuel .* abs(fipt - chpt) .* 6894.76);

ox_mdot   = sign(oipt - chpt) .* inj_cda_ox   .* 1e-4 .* ...
            sqrt(2 .* rho_ox   .* abs(oipt - chpt) .* 6894.76);

MR        = ox_mdot ./ fuel_mdot;

% Smoothed signals
fuel_mdot_smooth = movmean(fuel_mdot, window);
ox_mdot_smooth   = movmean(ox_mdot,   window);
MR_smooth        = movmean(MR,        window);

% Plot fuel/ox flows
figure;
plot(time, fuel_mdot_smooth, 'b-', 'LineWidth', 2); hold on;
plot(time, ox_mdot_smooth,   'r-', 'LineWidth', 2);
xlabel('Time (s)',  'FontSize', 15);
ylabel('Mass Flow (kg/s)', 'FontSize', 15);
title(sprintf(['Fuel and Oxidizer Mass Flow vs Time (Injector)\n', ...
    '(CdA_f = %.3f cm^2, CdA_{ox} = %.3f cm^2)'], inj_cda_fuel, inj_cda_ox), ...
    'FontSize', 15);
legend('Fuel Mdot', 'Oxidizer Mdot', 'Location', 'best');
grid on;

% Plot mixture ratio
figure;
plot(time, MR_smooth, 'k-', 'LineWidth', 2, ...
    'MarkerIndices', 1:50:length(time));
xlabel('Time (s)',  'FontSize', 15);
ylabel('Mixture Ratio (Injector)', 'FontSize', 15);
title(sprintf(['Mixture Ratio vs Time\n', ...
    '(CdA_f = %.3f cm^2, CdA_{ox} = %.3f cm^2)'], inj_cda_fuel, inj_cda_ox), ...
    'FontSize', 15);
grid on;

%% Runline Pressures
figure;
plot(time, movmean(frunpt, 50), 'b-', 'LineWidth', 1.5); hold on;
plot(time, movmean(fipt,   50), 'r-', 'LineWidth', 1.5);
xlabel('Time (s)',  'FontSize', 15);
ylabel('Pressure (psia)', 'FontSize', 15);
title('Fuel Runline vs Fuel Injector Pressure', 'FontSize', 15);
legend('FRUNPT', 'FIPT', 'Location', 'best');
grid on;

%% Flow Meter Derived Mass Flows
% LOX Venturi
mdot_ox_venturi = lox_venturi_cd * pi * (lox_venturi_inlet_id/(2*39.37))^2 .* ...
                  sqrt((2*rho_ox*abs(odp)*6894.76) ./ ...
                  ((lox_venturi_inlet_id / lox_venturi_throat_id)^4 - 1));

m_ox_cum   = cumtrapz(time, mdot_ox_venturi); 

% Fuel Flowmeter
mdot_fuel_meter = flowmeter * rho_fuel/1000 * flowmeter_multiplier;
m_fuel_cum      = cumtrapz(time, mdot_fuel_meter); 

% Plot flow rates and integrated mass
figure;
yyaxis left;
plot(time, mdot_ox_venturi,  'b-', 'LineWidth', 2); hold on;
plot(time, mdot_fuel_meter,  'r-', 'LineWidth', 2);
xlabel('Time (s)',  'FontSize', 15);
ylabel('Mass Flow (kg/s)', 'FontSize', 15);

yyaxis right;
plot(time, m_ox_cum,   'b--', 'LineWidth', 2);
plot(time, m_fuel_cum, 'r--', 'LineWidth', 2);
ylabel('Cumulative Mass Used (kg)', 'FontSize', 15);

title('LOX and Fuel Mass Flow & Cumulative Mass vs Time', 'FontSize', 15);
legend('LOX Mdot','Fuel Mdot','LOX Mass','Fuel Mass','Location','best');
grid on;

% Mixture ratio (meters)
MR_meters        = mdot_ox_venturi ./ mdot_fuel_meter;
MR_meters_smooth = movmean(MR_meters, window);

figure;
plot(time, MR_meters_smooth, 'k-', 'LineWidth', 2);
xlabel('Time (s)',  'FontSize', 15);
ylabel('Mixture Ratio', 'FontSize', 15);
title('Mixture Ratio vs Time (Flow Meters)', 'FontSize', 15);
grid on;

%% Pressurant Pressures
% Regulated pressurant pressure
figure;
plot(time, prpt2, 'k-', 'LineWidth', 2, ...
    'MarkerIndices', 1:50:length(time));
xlabel('Time (s)',  'FontSize', 15);
ylabel('Pressure (psia)', 'FontSize', 15);
title('Regulated Pressurant Pressure', 'FontSize', 15);
grid on;

% LOX-side pressurant pressure
figure;
plot(time, otpt,  'k-', 'LineWidth', 2, ...
    'MarkerIndices', 1:50:length(time)); hold on;
plot(time, prpto, 'r-', 'LineWidth', 2, ...
    'MarkerIndices', 1:50:length(time));
xlabel('Time (s)',  'FontSize', 15);
ylabel('Pressure (psia)', 'FontSize', 15);
title('LOX-Side Pressurant Pressure', 'FontSize', 15);
legend('OTPT','PRPTO','Location','best');
grid on;


