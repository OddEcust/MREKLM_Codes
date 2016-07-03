function [Res, t_train] = MSE_MultiClass(trainSet , testSet)
    
    totalClass = size(trainSet , 2) ; 
    tstLabel = testSet(:, end);
    testSet(:, end) = [];
    trainData = [];
    for i = 1 : totalClass
        tmp = trainSet{i};
        tmp(:, end+1) = i;
        trainData = [trainData; tmp];
    end
    trnLabel = trainData(:, end);
    lenTrn = length(trnLabel);
    trainData(:, end) = [];
    T = zeros(lenTrn, totalClass);
    labels = diag(ones(totalClass, 1));
    for i = 1 : totalClass
       ind = find(trnLabel == i);
       T(ind, :) = repmat(labels(i,:), length(ind), 1); 
    end
    %T = T.*2 -1;
    tic;
    Res = MSE_Fuc(trainData, T, [testSet, tstLabel]);
    t_train = toc;
end

function Res = MSE_Fuc(train, T, test)
    [lenTst, dim] = size(test);
    tstLabel = test(:, end);
    test(:, end) = [];
    [a, trnLabel] = max(T');
    trnLabel = trnLabel';    
    train(:, end+1) = 1;
    
    W = pinv(train)*T;
    
    trn_out = train* W;
    [a, trn_predict] = max(trn_out');
    trn_predict = trn_predict';
    trnReg = length(find(trn_predict == trnLabel))/length(trnLabel);
    
    test(:, end+1) = 1;
    tst_out = test* W;    
    [a, tst_predict] = max(tst_out') ;
    tst_predict = tst_predict' ;    
    tstReg = length(find(tst_predict == tstLabel))/lenTst ;
    
    Res.trn_Class = trn_predict;
    Res.trnReg = trnReg;    
    Res.tst_Class = tst_predict ;
    Res.tstReg = tstReg ;
end


