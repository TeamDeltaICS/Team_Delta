%% ============================================================
% uav_landing_paper.m  (VERSIÓN COMPLETA LISTA PARA EJECUTAR)
% Pipeline EXACTO del paper:
%  PhysModel → GRU Residual → Kalman → Fuzzy → Flare → Control
% =============================================================

clear; clc;

%% ------------------------------------------------------------
% 1) CONFIGURACIÓN CoppeliaSim + RemoteAPI
% -------------------------------------------------------------
addpath('C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\programming\zmqRemoteApi\clients\matlab');

client = RemoteAPIClient();
sim = client.getObject('sim');
sim.startSimulation();

quad_base = sim.getObject('/Quadcopter');
deck      = sim.getObject('/OmniPlatform');


%% ------------------------------------------------------------
% 2) TIEMPOS
% -------------------------------------------------------------
dt = 0.05;
t  = 0;


%% ------------------------------------------------------------
% 3) CARGA ONNX DEL GRU (si se puede)
% -------------------------------------------------------------
onnxFile = 'C:\Users\Enric\Documents\NTUST\ICS\Simulador\gru_paper.onnx';
useOnnx  = false;

if isfile(onnxFile)
    try
        net = importONNXNetwork(onnxFile, ...
            'OutputLayerType','regression', ...
            'ImportWeights',true);
        useOnnx = true;
        disp('ONNX GRU network loaded via MATLAB.');
    catch
        warning('MATLAB ONNX import failed. Will use Python fallback.');
    end
else
    warning('gru_paper.onnx not found. Using Python fallback.');
end


%% ------------------------------------------------------------
% 4) PARÁMETROS DEL MODELO FÍSICO DEL DECK (paper)
% -------------------------------------------------------------
physParams.Ax  = 0.0;
physParams.wx  = 0.0;
physParams.phx = 0.0;
physParams.Ay  = 0.0;
physParams.wy  = 0.0;
physParams.phy = 0.0;
physParams.Ah  = 0.0;
physParams.wz  = 0.0;
physParams.phz = 0.0;

physParams.vx = 0.0;
physParams.vy = 0.0;

deck0 = cell2mat(sim.getObjectPosition(deck, -1))';


%% ------------------------------------------------------------
% 5) KALMAN FILTER
% -------------------------------------------------------------
dt_k = dt;

F = eye(6);
F(1,2)=dt_k;
F(3,4)=dt_k;
F(5,6)=dt_k;

Q = eye(6)*1e-4;

H = [1 0 0 0 0 0;
     0 0 1 0 0 0;
     0 0 0 0 1 0];

R = eye(3)*1e-3;

xk = zeros(6,1);
Pk = eye(6)*1e-2;


%% ------------------------------------------------------------
% 6) FUZZY DECISIÓN (según paper)
% -------------------------------------------------------------
fuzzy_func = @(pvis, dnorm) fuzzy_mode_matlab(pvis, dnorm);


%% ------------------------------------------------------------
% 7) CONTROL (ganancias)
% -------------------------------------------------------------
mass = 1.5;
g    = 9.81;

Kp_xy = 1.5;
Kd_xy = 0.6;

Kp_z  = 2.0;
Kd_z  = 1.0;

max_att = 0.4;


%% ------------------------------------------------------------
% 8) HISTORIA DEL GRU (paper: dimensión 6)
%    [px, py, vx, vy, wind, mask]
% -------------------------------------------------------------
N      = 20;
in_dim = 6;
history = zeros(N, in_dim);


%% ------------------------------------------------------------
% 9) FLARE PROFILE (paper)
% -------------------------------------------------------------
inFlare        = false;
flareStartT    = 0;
flareDuration  = 1.2;
z_flare_start  = 0;


%% ------------------------------------------------------------
% 10) BUCLE PRINCIPAL
% -------------------------------------------------------------
max_time = 180;

while sim.getSimulationState() ~= sim.simulation_stopped  &&  t < max_time
    
    % ---------------------------------------------------------
    % (1) MEDIDA desde simulación (posición y velocidad)
    % ---------------------------------------------------------
    pos = cell2mat(sim.getObjectPosition(quad_base,-1));
    vel6 = cell2mat(sim.getObjectVelocity(quad_base));
    meas_pos = pos(:);
    
    % ---------------------------------------------------------
    % (2) KALMAN FILTER
    % ---------------------------------------------------------
    % predict
    xk = F*xk;
    Pk = F*Pk*F' + Q;
    
    % update
    z = meas_pos;
    S = H*Pk*H' + R;
    K = Pk*H'/S;
    xk = xk + K*(z - H*xk);
    Pk = (eye(6) - K*H)*Pk;
    
    
    % ---------------------------------------------------------
    % (3) PREDICCIÓN FÍSICA DEL DECK
    % ---------------------------------------------------------
    t_future = t + dt;
    p_phys = deck_predict(physParams, t_future, deck0);   % [x;y;z]
    phys_now_xy = p_phys(1:2);
    
    
    % ---------------------------------------------------------
    % (4) FEATURES para GRU según paper
    % ---------------------------------------------------------
    vel_est_xy = [xk(2); xk(4)];
    wind_now = 0.0;
    mask_now = 1.0;
    
    new_feat = [phys_now_xy(:); vel_est_xy(:); wind_now; mask_now]; % 6x1
    history = [history(2:end,:); new_feat(:)'];
    
    
    % ---------------------------------------------------------
    % (5) GRU INFERENCE (MATLAB ONNX → Python fallback)
    % ---------------------------------------------------------
    try_matlab = useOnnx;
    got_pred = false;
    
    if try_matlab
        try
            inp = permute(history, [2,1]);   % [6,20]
            ypred = predict(net, reshape(inp, [in_dim, N, 1]));
            y = squeeze(ypred);
            dx = double(y(1));
            dy = double(y(2));
            pvis = sigmoid(double(y(3)));
            logvar_x = double(y(4));
            logvar_y = double(y(5));
            got_pred = true;
        catch
            try_matlab = false;
        end
    end
    
    if ~try_matlab
        try
            py_mod = py.importlib.import_module('gru_infer');
            py_out = py_mod.infer(py.numpy.array(history));
            arr = double(py.array(py_out));
            dx = arr(1);
            dy = arr(2);
            pvis = arr(3);
            logvar_x = arr(4);
            logvar_y = arr(5);
            got_pred = true;
        catch
            got_pred = false;
        end
    end
    
    if ~got_pred
        dx=0; dy=0; pvis=1.0; logvar_x=log(1e-4); logvar_y=log(1e-4);
    end
    
    mean_res = [dx; dy; 0];
    logvar   = [logvar_x; logvar_y; log(1e-4)];
    
    
    % ---------------------------------------------------------
    % (6) PREDICCIÓN HÍBRIDA
    % ---------------------------------------------------------
    p_total = p_phys(:) + mean_res(:);
    
    
    % ---------------------------------------------------------
    % (7) DISTANCIA NORMALIZADA PARA FUZZY
    % ---------------------------------------------------------
    dz = xk(5) - deck0(3);
    dnorm = min(max((dz - 0.05)/1.0, 0),1);
    
    [mode, mode_val] = fuzzy_func(pvis, dnorm);
    
    
    % ---------------------------------------------------------
    % (8) CONTROL (incluye FLARE)
    % ---------------------------------------------------------
    pos_xy = [xk(1); xk(3)];
    vel_xy = [xk(2); xk(4)];
    
    switch mode
        
        case "DESCEND"
            dz_current = xk(5) - deck0(3);
            
            flare_trigger_h = 0.6;
            
            if ~inFlare && dz_current < flare_trigger_h
                inFlare = true;
                flareStartT = t;
                z_flare_start = xk(5);
            end
            
            if inFlare
                s = (t - flareStartT)/flareDuration;
                s = min(max(s,0),1);
                
                touchdown_offset = 0.02;
                
                z_ref = z_flare_start - ...
                       (z_flare_start - (deck0(3)+touchdown_offset)) * ...
                       (1 - cos(pi*s))/2;
                
                if s >= 1
                    inFlare = false;
                    if abs(xk(6)) < 0.3 && dz_current < 0.05
                        fprintf('>>> TOUCHDOWN at t=%.2f s\n', t);
                        sim.setFloatSignal('ext_thrust', 0.0);
                        sim.setFloatSignal('ext_roll',  0.0);
                        sim.setFloatSignal('ext_pitch', 0.0);
                        sim.setFloatSignal('ext_yaw',   0.0);
                        break;
                    end
                end
                
            else
                z_ref = p_total(3);
            end
            
            xy_ref = p_total(1:2);
            descend_gain = 1.0;
            
            
        case "APPROACH"
            z_ref = deck0(3) + 0.6;
            xy_ref = p_total(1:2);
            descend_gain = 0.2;
            
        case "HOLD"
            z_ref = xk(5);
            xy_ref = pos_xy;
            descend_gain = 0.0;
            
        case "ABORT"
            z_ref = xk(5) + 1.0;
            xy_ref = pos_xy + [-1;0];
            descend_gain = -0.5;
    end
    
    
    % ---------------------------------------------------------
    % (9) POS → ACC → ACTUADORES
    % ---------------------------------------------------------
    err_xy = xy_ref - pos_xy;
    err_v  = -vel_xy;
    acc_cmd_xy = Kp_xy*err_xy + Kd_xy*err_v;
    
    roll_sp  = -acc_cmd_xy(1)/g;
    pitch_sp =  acc_cmd_xy(2)/g;
    roll_sp  = max(min(roll_sp,  max_att), -max_att);
    pitch_sp = max(min(pitch_sp, max_att), -max_att);
    
    z_err = z_ref - xk(5);
    vz_err = -xk(6);
    a_z = Kp_z*z_err + Kd_z*vz_err + descend_gain;
    
    thrust_total = mass*(g + a_z);
    thrust_signal = thrust_total / (mass*(g+3));
    % compute thrust ratio relative to hover
    hover_total = mass * g;            % Newtons to hover (m*g)
    thrust_ratio = thrust_total / hover_total;   % 1.0 => hover
    
    % clip ratio to safe band
    min_ratio = 0.6;  % si baja de esto, va a caer en picado
    max_ratio = 1.8;  % límite superior
    thrust_ratio = max(min(thrust_ratio, max_ratio), min_ratio);
    
    % send normalized thrust_signal (1 => hover)
    sim.setFloatSignal('ext_thrust', double(thrust_ratio));
    
        
    sim.setFloatSignal('ext_roll',   double(roll_sp));
    sim.setFloatSignal('ext_pitch',  double(pitch_sp));
    sim.setFloatSignal('ext_yaw',    0.0);
    
    
    pause(dt);
    t = t + dt;
end


%% LIMPIEZA
sim.clearFloatSignal('ext_thrust');
sim.clearFloatSignal('ext_roll');
sim.clearFloatSignal('ext_pitch');
sim.clearFloatSignal('ext_yaw');
sim.stopSimulation();


%% ============================================================
% FUNCIONES AUXILIARES
% ============================================================

function p = deck_predict(params, t, deck0)
    t = double(t);
    x = deck0(1) + params.vx*t + params.Ax*sin(params.wx*t + params.phx);
    y = deck0(2) + params.vy*t + params.Ay*sin(params.wy*t + params.phy);
    z = deck0(3) + params.Ah*sin(params.wz*t + params.phz);
    p = [x; y; z];
end

function s = sigmoid(x)
    s = 1./(1+exp(-x));
end
