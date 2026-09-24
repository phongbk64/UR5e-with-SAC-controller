function [agent, trainingStats, cfg] = train_UR5_SAC(maxEpisodes)
% Huan luyen moi SAC agent cho UR5e va luu duy nhat bo ket qua final.
% Cach chay: train_UR5_SAC          % 350 episode
%            train_UR5_SAC(200)     % so episode tuy chon

if nargin < 1, maxEpisodes = 350; end
validateattributes(maxEpisodes,{'numeric'},{'scalar','integer','positive'});

rootFolder = fileparts(mfilename('fullpath'));
resultFolder = fullfile(rootFolder,'final_training');
checkpointFolder = fullfile(resultFolder,'checkpoints');
if exist(resultFolder,'dir'), rmdir(resultFolder,'s'); end
if ~exist(checkpointFolder,'dir'), mkdir(checkpointFolder); end

[agent, env, cfg] = init_UR5_SAC();
modelName = 'UR5_SAC_Sim';
trainingOptions = rlTrainingOptions( ...
    'MaxEpisodes',maxEpisodes,'MaxStepsPerEpisode',cfg.Nmax, ...
    'StopTrainingCriteria','EpisodeCount','StopTrainingValue',maxEpisodes, ...
    'SaveAgentCriteria','EpisodeCount','SaveAgentValue',5, ...
    'SaveAgentDirectory',checkpointFolder,'Verbose',true,'Plots','none');
save(fullfile(resultFolder,'configuration_final.mat'),'cfg','trainingOptions');

fprintf('\nFINAL TRAINING: %d EPISODES\n',maxEpisodes);
try
    trainingStats = train(agent,env,trainingOptions);
    set_param(modelName,'FastRestart','off');
catch ME
    set_param(modelName,'FastRestart','off');
    save(fullfile(resultFolder,'interrupted_agent.mat'),'agent','cfg','-v7.3');
    rethrow(ME);
end

T = table(trainingStats.EpisodeIndex(:),trainingStats.EpisodeReward(:), ...
    trainingStats.EpisodeSteps(:), ...
    'VariableNames',{'Episode','Reward','Steps'});
writetable(T,fullfile(resultFolder,'training_final.csv'));

f = figure('Visible','off','Color','w');
plot(T.Episode,T.Reward,'Color',[0.65 0.75 0.90]); hold on;
plot(T.Episode,movmean(T.Reward,min(10,height(T))),'b','LineWidth',1.6);
xlabel('Episode'); ylabel('Reward'); grid on;
legend('Episode reward','Moving average','Location','best');
title('UR5 SAC final training');
exportgraphics(f,fullfile(resultFolder,'training_curve_final.png'),'Resolution',180);
close(f);

% Danh gia deterministic tung checkpoint va chon mang dat nguong tot nhat.
files = dir(fullfile(checkpointFolder,'Agent*.mat'));
episodes = zeros(numel(files),1);
for k = 1:numel(files)
    episodes(k) = sscanf(files(k).name,'Agent%d.mat');
end
[episodes,order] = sort(episodes); files = files(order);
n = numel(files);
posErr = zeros(n,1); angleErr = zeros(n,1); steps = zeros(n,1);
reached = false(n,1);
for k = 1:n
    C = load(fullfile(files(k).folder,files(k).name));
    if isfield(C,'saved_agent'), candidate = C.saved_agent; else, candidate = C.agent; end
    if isprop(candidate,'UseExplorationPolicy'), candidate.UseExplorationPolicy = false; end
    assignin('base','agent',candidate);
    experience = sim(env,candidate,rlSimulationOptions('MaxSteps',cfg.Nmax));
    obs = squeeze(experience.Observation.UR5Observations.Data);
    if size(obs,1) ~= 15, obs = obs.'; end
    ep = obs(7:9,:)./cfg.ObsScale(7:9);
    vef = obs(10:12,:)./cfg.ObsScale(10:12);
    vg = obs(13:15,:)./cfg.ObsScale(13:15);
    d = vecnorm(ep,2,1);
    c = sum(vef.*vg,1)./(vecnorm(vef,2,1).*vecnorm(vg,2,1));
    a = acos(max(-1,min(1,c)));
    posErr(k) = d(end); angleErr(k) = a(end); steps(k) = numel(d)-1;
    reached(k) = posErr(k) <= cfg.eps_d && angleErr(k) <= cfg.eps_a;
    fprintf('Checkpoint %d: %.4f cm, %.4f deg, success=%d\n', ...
        episodes(k),100*posErr(k),rad2deg(angleErr(k)),reached(k));
end

score = posErr/cfg.eps_d + angleErr/cfg.eps_a;
if any(reached)
    valid = find(reached);
    [~,j] = min(score(valid) + 1e-4*steps(valid)); best = valid(j);
else
    evaluationTable = table(episodes,100*posErr,rad2deg(angleErr),steps,reached,score, ...
        'VariableNames',{'Episode','PositionErrorCm','OrientationErrorDeg', ...
        'Steps','GoalReached','NormalizedErrorScore'});
    writetable(evaluationTable,fullfile(resultFolder,'checkpoint_evaluation.csv'));
    error('Khong checkpoint nao dat nguong 0.5 cm va 1 do.');
end
C = load(fullfile(files(best).folder,files(best).name));
if isfield(C,'saved_agent'), agent = C.saved_agent; else, agent = C.agent; end
selectedEpisode = episodes(best);
save(fullfile(resultFolder,'trained_agent_final.mat'), ...
    'agent','trainingStats','cfg','selectedEpisode','-v7.3');

evaluationTable = table(episodes,100*posErr,rad2deg(angleErr),steps,reached,score, ...
    'VariableNames',{'Episode','PositionErrorCm','OrientationErrorDeg', ...
    'Steps','GoalReached','NormalizedErrorScore'});
writetable(evaluationTable,fullfile(resultFolder,'checkpoint_evaluation.csv'));
finalEvaluation = evaluationTable(best,:);
writetable(finalEvaluation,fullfile(resultFolder,'evaluation_final.csv'));

if exist(checkpointFolder,'dir'), rmdir(checkpointFolder,'s'); end
fprintf('Da chon checkpoint episode %d. Ket qua final: %s\n', ...
    selectedEpisode,resultFolder);
end
