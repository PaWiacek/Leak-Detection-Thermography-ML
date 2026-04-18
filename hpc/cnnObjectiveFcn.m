function objective = cnnObjectiveFcn(optVars)

% === BEST OBJECTIVE INITIALIZATION ===
persistent bestObjective
if isempty(bestObjective)
    bestObjective = Inf;
end

% === CONFIGURATION ===
baseDir = fullfile('..', 'OK_dataset2026_color_train_aug');

dataDirTrain = fullfile(baseDir, 'Train');
dataDirVal   = fullfile(baseDir, 'Validation');
dataDirTest  = fullfile(baseDir, 'Test');
inputSize = [640 480 3];

% === LOAD DATA ===
imdsTrain = imageDatastore(dataDirTrain, "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");
imdsValidation = imageDatastore(dataDirVal, "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");
imdsTest = imageDatastore(dataDirTest, "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

% === DISPLAY CLASS DISTRIBUTIONS ===
disp('Class distribution in TRAIN:');
disp(countEachLabel(imdsTrain));

disp('Class distribution in VALIDATION:');
disp(countEachLabel(imdsValidation));

disp('Class distribution in TEST:');
disp(countEachLabel(imdsTest));

% === RESIZE (NO AUGMENTATION) ===
augImdsTrain = augmentedImageDatastore(inputSize, imdsTrain);
augImdsValidation= augmentedImageDatastore(inputSize, imdsValidation);
augImdsTest = augmentedImageDatastore(inputSize, imdsTest);

        % === BUILD NETWORK ===
	layers = createCNN(inputSize, ...
	optVars.FilterSize1, optVars.NumFilters1, ...
	optVars.FilterSize2, optVars.NumFilters2, ...
	optVars.StrideVal1, optVars.StrideVal2, ...
	optVars.FCLayerSize, optVars.Dropout);

	
        % === TRAINING OPTIONS ===
 	valFreq = floor(numel(imdsTrain.Files)/optVars.MiniBatchSize)
	options = trainingOptions('adam', ...
		'InitialLearnRate', optVars.InitialLearnRate, ...
		'MaxEpochs', 100, ...
		'MiniBatchSize', optVars.MiniBatchSize, ...
		'Shuffle', 'every-epoch', ...
		'ValidationData', augImdsValidation, ...
		'ValidationFrequency', valFreq, ...
		'ValidationPatience', 10, ...
		'Verbose', false, ...
		'Plots', 'none');


        % === TRAIN ===
        net = trainNetwork(augImdsTrain, layers, options);

        % === VALIDATION ACCURACY ===
        YPred = classify(net, augImdsValidation);  % FIXED
        YVal  = imdsValidation.Labels;             % FIXED
        accuracy = mean(YPred == YVal);

        % === RETURN OBJECTIVE ===
        objective = 1 - accuracy;

	% === BEST OBJECTIVE SAVE
	lockFile = 'bestNet.lock';
	bestFile = 'bestNet.mat';

	if ~exist(lockFile,'file')

		fid = fopen(lockFile,'w');
		if fid ~= -1
			fclose(fid);

			doSave = true;

			if exist(bestFile,'file')
				S = load(bestFile,'objective');
				if objective >= S.objective
					doSave = false;
				end
			end

			if doSave
				save(bestFile,'net','objective','optVars');
			end

			delete(lockFile);
		end
	end

end
