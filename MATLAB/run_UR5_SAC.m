function results = run_UR5_SAC()
% Chay deterministic agent final, ve ket qua va xuat quy dao khop ra Excel.

rootFolder = fileparts(mfilename('fullpath'));
resultFolder = fullfile(rootFolder,'final_training');
agentFile = fullfile(resultFolder,'trained_agent_final.mat');
assert(isfile(agentFile), ...
    'Khong tim thay trained_agent_final.mat. Hay chay train_UR5_SAC truoc.');

S = load(agentFile,'agent','cfg');
agent = S.agent; cfg = S.cfg;
if isprop(agent,'UseExplorationPolicy'), agent.UseExplorationPolicy = false; end
assignin('base','agent',agent);

modelName = 'UR5_SAC_Sim';
load_system(fullfile(rootFolder,modelName + ".slx"));

obsInfo = rlNumericSpec([15 1]);
actInfo = rlNumericSpec([6 1], ...
    'LowerLimit',-cfg.dqMax(:),'UpperLimit',cfg.dqMax(:));
env = rlSimulinkEnv(modelName,modelName + "/RL Agent",obsInfo,actInfo);
env.UseFastRestart = 'on';
rng(0,'twister');
experience = sim(env,agent,rlSimulationOptions('MaxSteps',cfg.Nmax));
set_param(modelName,'FastRestart','off');

obsNames = fieldnames(experience.Observation);
obsSeries = experience.Observation.(obsNames{1});
obs = squeeze(obsSeries.Data);
if size(obs,1) ~= 15, obs = obs.'; end
time = obsSeries.Time(:);
q = obs(1:6,:)*2*pi;
ep = obs(7:9,:);
vef = obs(10:12,:);
vg = obs(13:15,:);
pe = cfg.pg(:)-ep;
positionError = vecnorm(ep,2,1);
cosAngle = sum(vef.*vg,1)./(vecnorm(vef,2,1).*vecnorm(vg,2,1));
orientationErrorDeg = rad2deg(acos(max(-1,min(1,cosAngle))));

f1=figure('Name','UR5 joint variables','Color','w');
plot(time,q.','LineWidth',1.2); grid on;
xlabel('Time (s)'); ylabel('Joint angle (rad)');
legend('q_1','q_2','q_3','q_4','q_5','q_6','Location','best');
title('Joint variables over time');
exportgraphics(f1,fullfile(resultFolder,'joint_variables.png'),'Resolution',180);

f2=figure('Name','UR5 end-effector pose','Color','w');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');
nexttile; plot(time,pe.','LineWidth',1.2); hold on;
yline(cfg.pg(1),'--'); yline(cfg.pg(2),'--'); yline(cfg.pg(3),'--');
grid on; ylabel('Position (m)'); title('End-effector coordinates');
legend('x','y','z','x goal','y goal','z goal','Location','best');
nexttile; plot(time,vef.','LineWidth',1.2); hold on;
yline(cfg.vg(1),'--'); yline(cfg.vg(2),'--'); yline(cfg.vg(3),'--');
grid on; xlabel('Time (s)'); ylabel('Direction component');
title('End-effector z-direction');
legend('v_x','v_y','v_z','v_{gx}','v_{gy}','v_{gz}','Location','best');
exportgraphics(f2,fullfile(resultFolder,'end_effector_pose.png'),'Resolution',180);

f3=figure('Name','UR5 tracking errors','Color','w');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');
nexttile; plot(time,100*positionError,'b','LineWidth',1.4); hold on;
yline(100*cfg.eps_d,'--r','0.5 cm threshold'); grid on;
ylabel('Position error (cm)'); title('Tracking errors');
nexttile; plot(time,orientationErrorDeg,'m','LineWidth',1.4); hold on;
yline(rad2deg(cfg.eps_a),'--r','1 deg threshold'); grid on;
xlabel('Time (s)'); ylabel('Orientation error (deg)');
exportgraphics(f3,fullfile(resultFolder,'tracking_errors.png'),'Resolution',180);

trajectory = array2table([time q.'], ...
    'VariableNames',{'Time_s','q1_rad','q2_rad','q3_rad', ...
    'q4_rad','q5_rad','q6_rad'});
excelFile = fullfile(resultFolder,'joint_trajectory.xlsx');
writetable(trajectory,excelFile,'Sheet','JointTrajectory');

goalReached = positionError(end)<=cfg.eps_d && ...
    orientationErrorDeg(end)<=rad2deg(cfg.eps_a);
results = struct('time',time,'q',q,'position',pe,'direction',vef, ...
    'positionError',positionError,'orientationErrorDeg',orientationErrorDeg, ...
    'goalReached',goalReached,'excelFile',excelFile);
save(fullfile(resultFolder,'execution_results.mat'),'results');
set_param(modelName,'Dirty','off');

fprintf('\nFINAL EXECUTION\n');
fprintf('Position error    : %.4f cm\n',100*positionError(end));
fprintf('Orientation error : %.4f deg\n',orientationErrorDeg(end));
fprintf('Goal reached      : %d\n',goalReached);
fprintf('Excel trajectory  : %s\n',excelFile);
end
