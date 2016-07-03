function Res = GMKSVM_MultiClass(trainSet, testSet, conf)
%
% train = {c1, c2, ..., cn}
% c = [X, label]
% test =  [X, label]
% conf = {C, kNum, kType, kPar}
%

totalClass = size(trainSet , 2) ;
[lenTest , dim] = size(testSet) ;
testLable = testSet(:,dim) ;
resultMat = zeros(lenTest , totalClass) ;
t_train = 0 ;
% forSave = cell(totalClass, totalClass);
for i = 1 : totalClass
    classOne = trainSet{i}(:, 1:dim-1) ;
    for j = i +1 : totalClass
        classTwo = trainSet{j}(:, 1:dim-1) ;
        [temp, t]= GMKSVM_OAO(classOne , classTwo , testSet(:,1:dim-1), conf) ;
        t_train = t_train + t ;
        % forSave{i,j} = temp;
        tLabel = temp.finalLabel;
        clear temp;
        indexClassOne = find(tLabel == 1);
        resultMat(indexClassOne , i) = resultMat(indexClassOne , i) + 1 ;
        indexClassTwo = find(tLabel == -1) ;
        resultMat(indexClassTwo , j) = resultMat(indexClassTwo , j) + 1 ;
        clear classTwo;
    end
end
[C, finalClass] = max((resultMat')) ;
regRate = size(find(finalClass' == testLable),1)/lenTest ;
% Res.forSave = forSave;
Res.finalClass = finalClass;
Res.t_train = t_train;
Res.regRate = regRate;
end

function [forSave, t] = GMKSVM_OAO(classOne, classTwo, test, conf)
%
% classOne, classTwo without label
% test without label
% conf = {C, kNum, kType, kPar}
% if out > 0, label = i, else label = j
%

    parameters = gmksvm_parameter();
    parameters.C = conf.C;
    kNum = conf.kNum;
    
    for kId = 1 : kNum
       if strcmp(conf.kType{kId}, 'g')
           kPar = conf.Delta(kId)* conf.kPar;
           conf.kType{kId} = ['g', num2str(kPar)];
       end
    end
    
    parameters.ker = conf.kType;
    parameters.nor.dat = 'true';
    parameters.nor.ker = 'true';
    parameters.opt = 'smo'; %set to "libsvm" or "mosek" to change the optimizer
    
    % Reformulate the data for training %
    data = [classOne; classTwo];
    trnStruct.y = [1*ones(size(classOne, 1),1); -1*ones(size(classTwo,1),1)];
    trnStruct.X = data;
    trnStruct.ind = [1:size(data,1)]';  
    
    
    training_data = cell(1, kNum);
    testing_data = training_data;
    for kId = 1 : kNum
       training_data{kId} = trnStruct;  
       testing_data{kId}.X = test;
    end
    
    tic;
    model = gmksvm_train(training_data, parameters);
    t = toc;    % evaluate the training time
    
    output = gmksvm_test(testing_data, model);
    finalLabel = sign(output.dis);    
    kWeight = zeros(kNum,1);
    for kId = 1 : kNum
        kWeight(kId) = model.sup{kId}.eta;
    end
    
    forSave.output = output.dis;
    forSave.finalLabel = finalLabel;
    forSave.kWeight = kWeight;
    forSave.model = model;
end

function par=aveRBFPar(data , size)
    mat_temp = sum(data.^2,2) * ones(1,size) + ones(size,1)*sum(data.^2,2)' - 2* data*data';
    tempMean = 1/size^2 * sum(sum(mat_temp,1),2) ;
    par = sqrt(tempMean) ;
end
