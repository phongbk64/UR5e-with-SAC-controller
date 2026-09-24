function [agent, env, cfg] = init_UR5_SAC()
% Khoi tao SAC agent va moi truong UR5e tu UR5_SAC_Sim.slx.
% Cac tham so robot, goal, reward va nguong dung nam trong Model Workspace.

rootFolder = fileparts(mfilename('fullpath'));
modelName = 'UR5_SAC_Sim';
load_system(fullfile(rootFolder, modelName + ".slx"));
modelWS = get_param(modelName, 'ModelWorkspace');

names = {'Ts','Nmax','q0','pg','vg','qMin','qMax','dqMax', ...
    'wd','wa','eps_d','eps_a','BaseTransform','FixedTransforms', ...
    'JointAxes','ToolTransform','GeometryID','ProgressWeight', ...
    'AngularProgressWeight','SuccessBonus','InitialEd','InitialEa','ObsScale'};
for k = 1:numel(names)
    cfg.(names{k}) = getVariable(modelWS, names{k});
end
cfg.q0 = cfg.q0(:); cfg.qMin = cfg.qMin(:); cfg.qMax = cfg.qMax(:);
cfg.dqMax = cfg.dqMax(:); cfg.pg = cfg.pg(:); cfg.vg = cfg.vg(:)/norm(cfg.vg);
assert(all(cfg.q0 >= cfg.qMin & cfg.q0 <= cfg.qMax), 'q0 vuot gioi han khop.');

obsInfo = rlNumericSpec([15 1]);
obsInfo.Name = 'UR5 observations';
actInfo = rlNumericSpec([6 1], 'LowerLimit', -cfg.dqMax, ...
    'UpperLimit', cfg.dqMax);
actInfo.Name = 'Joint velocities';

rng(0, 'twister');
nObs = 15; nAct = 6; nHidden = 256;

actorGraph = layerGraph([
    featureInputLayer(nObs,'Normalization','none','Name','obs')
    fullyConnectedLayer(nHidden,'Name','actor_fc1')
    reluLayer('Name','actor_relu1')
    fullyConnectedLayer(nHidden,'Name','actor_fc2')
    reluLayer('Name','actor_relu2')]);
actorGraph = addLayers(actorGraph, fullyConnectedLayer(nAct,'Name','mean'));
actorGraph = addLayers(actorGraph, [
    fullyConnectedLayer(nAct,'Name','std_fc')
    softplusLayer('Name','std')]);
actorGraph = connectLayers(actorGraph,'actor_relu2','mean');
actorGraph = connectLayers(actorGraph,'actor_relu2','std_fc');
actor = rlContinuousGaussianActor(dlnetwork(actorGraph),obsInfo,actInfo, ...
    'ObservationInputNames','obs','ActionMeanOutputNames','mean', ...
    'ActionStandardDeviationOutputNames','std');

for c = 1:2
    criticGraph = layerGraph();
    criticGraph = addLayers(criticGraph, ...
        featureInputLayer(nObs,'Normalization','none','Name','obs'));
    criticGraph = addLayers(criticGraph, ...
        featureInputLayer(nAct,'Normalization','none','Name','act'));
    criticGraph = addLayers(criticGraph, [
        concatenationLayer(1,2,'Name','concat')
        fullyConnectedLayer(nHidden,'Name','critic_fc1')
        reluLayer('Name','critic_relu1')
        fullyConnectedLayer(nHidden,'Name','critic_fc2')
        reluLayer('Name','critic_relu2')
        fullyConnectedLayer(1,'Name','QValue')]);
    criticGraph = connectLayers(criticGraph,'obs','concat/in1');
    criticGraph = connectLayers(criticGraph,'act','concat/in2');
    critics(c) = rlQValueFunction(dlnetwork(criticGraph),obsInfo,actInfo, ...
        'ObservationInputNames','obs','ActionInputNames','act'); %#ok<AGROW>
end

agentOptions = rlSACAgentOptions('SampleTime',cfg.Ts, ...
    'DiscountFactor',0.99,'TargetSmoothFactor',0.005, ...
    'ExperienceBufferLength',1e6,'MiniBatchSize',256, ...
    'NumWarmStartSteps',1000);
optimizer = rlOptimizerOptions('LearnRate',1e-4,'GradientThreshold',1);
agentOptions.ActorOptimizerOptions = optimizer;
agentOptions.CriticOptimizerOptions = [optimizer optimizer];
agentOptions.EntropyWeightOptions.EntropyWeight = 0.01;
agentOptions.EntropyWeightOptions.LearnRate = 0;
agent = rlSACAgent(actor, critics, agentOptions);
assignin('base','agent',agent);

env = rlSimulinkEnv(modelName,modelName + "/RL Agent",obsInfo,actInfo);
env.UseFastRestart = 'on';
validateEnvironment(env);

fprintf('Khoi tao thanh cong: %s\n', cfg.GeometryID);
fprintf('Goal [m] = [%.6f %.6f %.6f], nguong = %.3f cm va %.3f deg\n', ...
    cfg.pg, 100*cfg.eps_d, rad2deg(cfg.eps_a));
end
