function layers = createCNN(inputSize, FilterSize1, NumFilters1, FilterSize2, NumFilters2, StrideVal1, StrideVal2, FCLayerSize, Dropout)

    layers = [
        imageInputLayer(inputSize, 'Name', 'input')

        convolution2dLayer(FilterSize1, NumFilters1, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        maxPooling2dLayer(2, 'Stride', StrideVal1, 'Name', 'pool1')

        convolution2dLayer(FilterSize2, NumFilters2, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer(2, 'Stride', StrideVal2, 'Name', 'pool2')

        fullyConnectedLayer(FCLayerSize, 'Name', 'fc1')
        reluLayer('Name', 'relu3')
        dropoutLayer(Dropout, 'Name', 'dropout')

        fullyConnectedLayer(2, 'Name', 'fc2')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')

    ];
end
