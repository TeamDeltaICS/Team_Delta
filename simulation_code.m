%% ============================================================
% simluation_code.m  
% WITH FINAL LANDING (FLARE MANEUVER)
% =============================================================
clear; clc;

%% 1) CONFIGURATION
addpath('C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\programming\zmqRemoteApi\clients\matlab');
client = RemoteAPIClient();
sim = client.getObject('sim');
sim.startSimulation();

quad_base   = sim.getObject('/Quadcopter');
quad_target = sim.getObject('/target');     
deck        = sim.getObject('/OmniPlatform');

[cite_start]%% 2) PHYSICAL PARAMETERS (Paper Eq 1-3) [cite: 83]
physParams.Ax = 0.20; physParams.wx = 0.6; physParams.phx = 0;
physParams.Ay = 0.15; physParams.wy = 0.5; physParams.phy = 0;
physParams.Ah = 0.10; physParams.wz = 0.7; physParams.phz = 0;
physParams.vx = 0.02; physParams.vy = -0.01;

%% 3) STATE INITIALIZATION
deck0 = cell2mat(sim.getObjectPosition(deck,-1))';
uav0  = cell2mat(sim.getObjectPosition(quad_base,-1))';

% Kalman State: [x, vx, y, vy, z, vz] initialized to the DECK's position
xk = [deck0(1); 0; deck0(2); 0; deck0(3); 0];
Pk = eye(6) * 0.1;

% Kalman Parameters
dt = 0.05;
F = eye(6); F(1,2)=dt; F(3,4)=dt; F(5,6)=dt;
Q = eye(6)*1e-4; H = [1 0 0 0 0 0; 0 0 1 0 0 0; 0 0 0 0 1 0]; R = eye(3)*1e-3;

%% 4) GRU HISTORY & FLARE VARIABLES
N = 20;
history = zeros(N, 6); 
for i=1:N
    % Pre-fill history with static initial deck state to avoid pvis=0 (cold start)
    history(i,:) = [deck0(1), deck0(2), 0, 0, 0, 1];
end

% --- NEW VARIABLES FOR FLARE ---
isFlaring = false;       % Are we currently executing the landing maneuver?
flareStartTime = 0;
flareDuration = 2.5;     % Seconds to touch the deck surface
flareStartOffset = 0;    % Relative height when flare starts
hasLanded = false;

%% 5) ONNX LOADING
onnxFile = 'gru_paper.onnx'; 
useOnnx = false;
if isfile(onnxFile)
    try
        net = importONNXNetwork(onnxFile, 'OutputLayerType','regression');
        useOnnx = true;
    catch, warning('ONNX failed.'); 
    end
end

%% ============================================================
% MAIN LOOP
% ============================================================
max_time = 180;
t = 0;
disp('Starting Control with Flare...');

while sim.getSimulationState() ~= sim.simulation_stopped  &&  t < max_time
    
    %% (1) SENSORS
    % Simulating Vision: Reading the true deck position
    real_deck_pos = cell2mat(sim.getObjectPosition(deck,-1))';
    % Reading UAV position (for relative distance)
    uav_pos       = cell2mat(sim.getObjectPosition(quad_base,-1))';
    
    %% (2) KALMAN UPDATE
    xk = F*xk; Pk = F*Pk*F' + Q;
    K = Pk*H'/(H*Pk*H' + R);
    xk = xk + K*(real_deck_pos - H*xk);
    Pk = (eye(6)-K*H)*Pk;
    
    %% (3) PHYSICAL PREDICTION
    pred_horizon = dt * 2.0; 
    t_future = t + pred_horizon;
    
    [cite_start]% Hybrid Model: Kalman Trend + Model Oscillation [cite: 83]
    osc_z = physParams.Ah * sin(physParams.wz * t_future + physParams.phz);
    
    p_phys_x = xk(1) + xk(2)*pred_horizon; % X Trend (Kalman estimated)
    p_phys_y = xk(3) + xk(4)*pred_horizon; % Y Trend
    p_phys_z = xk(5) + osc_z;              % Z Trend (Kalman base + Wave offset)
    
    p_total = [p_phys_x; p_phys_y; p_phys_z];
    
    %% (4) GRU / AI
    % Update history with the current estimated DECK state
    new_feat = [xk(1), xk(3), xk(2), xk(4), 0, 1]; 
    history = [history(2:end,:); new_feat];
    
    pvis = 0.5; % Default value
    if useOnnx, pvis = 0.9; end
    if t < 3.0, pvis = 1.0; end % Temporary confidence boost for Warmup
    
    %% (5) CONTROL LOGIC (FLARE VS FUZZY)
    % Relative vertical distance (UAV Z - Predicted Deck Z)
    dist_z = uav_pos(3) - p_total(3);
    
    % -- FLARE LOGIC --
    if hasLanded
        mode = "LANDED";
        mode_idx = 5;
        % Stick to the deck (slightly negative offset to ensure contact)
        target_ref = [p_total(1); p_total(2); p_total(3) - 0.2]; 
        
    elseif isFlaring
        mode = "FLARE";
        mode_idx = 6;
        
        % Calculate flare time
        t_elapsed = t - flareStartTime;
        progress = min(t_elapsed / flareDuration, 1.0);
        
        % Descent ramp: Current offset down to 0 (or slightly negative)
        % This offset is added to p_total(3), ensuring wave synchronization during descent.
        current_offset = flareStartOffset * (1 - progress);
        
        % If ramp finished, mark landed
        if progress >= 1.0
            hasLanded = true;
            disp('--- TOUCHDOWN ---');
        end
        
        target_ref = [p_total(1); p_total(2); p_total(3) + current_offset];
        
    else
        % -- NORMAL FUZZY LOGIC --
        % Normalized distance: 1 = Far/Safe, 0 = Close/Danger
        dnorm = min(max((dist_z - 0.1)/1.5, 0), 1);
        [mode, mode_idx] = fuzzy_logic_corrected(pvis, dnorm);
        
        % Check FLARE TRIGGER (Override Fuzzy for final commitment)
        % If mode is APPROACH/DESCEND, highly confident, and close enough
        if (strcmp(mode, "APPROACH") || strcmp(mode, "DESCEND")) && dist_z < 1.2 && pvis > 0.7
            isFlaring = true;
            flareStartTime = t;
            flareStartOffset = dist_z; % Start height for the ramp
            disp('--- INICIATING FLARE ---');
        end
        
        % Normal Fuzzy Setpoints
        switch mode
            case "DESCEND"
                target_ref = p_total; 
            case "APPROACH"
                z_hover = p_total(3) + 0.8; % Hold 80cm above the wave
                target_ref = [p_total(1); p_total(2); z_hover];
            case "HOLD"
                target_ref = [xk(1); xk(3); uav_pos(3)];
            case "ABORT"
                safe_altitude = deck0(3) + 2.5; % Fixed safe height
                target_ref = [xk(1); xk(3); safe_altitude];
        end
    end
    
    %% (6) MOVE TARGET
    curr_target = cell2mat(sim.getObjectPosition(quad_target,-1))';
    vec = target_ref - curr_target;
    
    % Adjust speed limit for Flare maneuver
    if isFlaring, speed_limit = 2.0; else, speed_limit = 1.0; end
    
    max_step = speed_limit * dt;
    if norm(vec) > max_step
        vec = vec * (max_step / norm(vec));
    end
    sim.setObjectPosition(quad_target, -1, (curr_target + vec)');
    
    %% DEBUG
    if mod(t, 0.5) < dt
        fprintf('T:%.1f | Mode:%s | DeckZ:%.2f | UAV_Z:%.2f | Offset:%.2f\n', ...
            t, mode, real_deck_pos(3), uav_pos(3), dist_z);
    end
    
    pause(dt);
    t = t + dt;
end
sim.stopSimulation();

%% FUZZY FUNCTION (Identical to V3 logic)
function [s_str, idx] = fuzzy_logic_corrected(pvis, d)
    % Based on Paper Table I: Pvis vs D (Distance/Clearance)
    TH_LOW_P = 0.4; TH_HI_P = 0.7;
    TH_LOW_D = 0.3; TH_HI_D = 0.8;
    
    if pvis < TH_LOW_P 
        if d < TH_LOW_D, s_str="ABORT"; idx=1; else, s_str="HOLD"; idx=2; end
    elseif pvis < TH_HI_P
        if d < TH_LOW_D, s_str="ABORT"; idx=1; else, s_str="APPROACH"; idx=3; end
    else
        if d < TH_LOW_D, s_str="ABORT"; idx=1; 
        elseif d > TH_HI_D, s_str="DESCEND"; idx=4; 
        else, s_str="APPROACH"; idx=3; 
        end
    end
    
    % Override to descend fast if too high (helps with initial engagement)
    if d > 0.9
         s_str="APPROACH"; idx=3;
         if pvis > 0.8, s_str="DESCEND"; idx=4; end
    end
end