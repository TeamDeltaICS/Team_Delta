function [mode, mode_val] = fuzzy_mode_matlab(pvis, d)
% --------------------------------------------------------------
% fuzzy_mode_matlab
% Implementación EXACTA del fuzzy supervisor del paper.
% --------------------------------------------------------------

%% 1) FUNCIONES DE MEMBRESÍA

% pvis membership
pvis_low    = trapmf(pvis, [0 0 0.25 0.40]);
pvis_med    = trimf(pvis, [0.30 0.50 0.70]);
pvis_high   = trapmf(pvis, [0.60 0.75 1 1]);

% distancia d membership
d_far   = trapmf(d, [0.5 0.7 1 1]);
d_mid   = trimf(d, [0.2 0.45 0.7]);
d_close = trapmf(d, [0 0 0.25 0.45]);

%% 2) RULES (del paper)

r_abort    = pvis_low;
r_hold     = min(pvis_med, d_far);
r_approach = min(pvis_high, d_mid);
r_descend  = min(pvis_high, d_close);

%% 3) MAX RULE

[mode_val, idx] = max([r_abort, r_hold, r_approach, r_descend]);
modes = ["ABORT","HOLD","APPROACH","DESCEND"];
mode = modes(idx);

end


%% ============================
% MEMBERSHIP FUNCTIONS FIXED
%% ============================

function y = trimf(x, params)
a = params(1); b = params(2); c = params(3);
y = max(min([ (x-a)/(b-a), (c-x)/(c-b) ]), 0);
end

function y = trapmf(x, params)
a = params(1); b = params(2); c = params(3); d = params(4);
y = max(min([ (x-a)/(b-a), 1, (d-x)/(d-c) ]), 0);
end
