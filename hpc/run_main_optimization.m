
% Define hyperparameters to optimize

params = [
    optimizableVariable('FilterSize1', [3 7], 'Type', 'integer')
    optimizableVariable('NumFilters1', [8 16], 'Type', 'integer')
    optimizableVariable('FilterSize2', [3 7], 'Type', 'integer')
    optimizableVariable('NumFilters2', [16 64], 'Type', 'integer')
    optimizableVariable('StrideVal1', [1 3], 'Type', 'integer')
    optimizableVariable('StrideVal2', [1 3], 'Type', 'integer')
    optimizableVariable('FCLayerSize', [16 64], 'Type', 'integer')
    optimizableVariable('Dropout', [0.2 0.5], 'Type', 'real')
    optimizableVariable('InitialLearnRate', [1e-6, 5e-4], 'Transform', 'log')
    optimizableVariable('MiniBatchSize', [16, 24], 'Type', 'integer')

];

% Start parallel pool (if not already running)

if isempty(gcp('nocreate'))
    parpool('local', 24);  % Use up to the number of CPUs requested in SLURM
end


% Run Bayesian optimization with parallel evaluation

results = bayesopt(@cnnObjectiveFcn, params, ...
    'MaxObjectiveEvaluations', 200, ...
    'Verbose', 1, ...
    'UseParallel', true, ...
    'PlotFcn', {@plotMinObjective});