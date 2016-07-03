warning off ;
clear;
clc;

load('..\0 Datasets\twonorm_v');
dataSet = twonorm_v.data;
kPar = twonorm_v.kPar;
clear twonorm_v;
saveMatName = '.\report\twonorm_v.mat';
saveFileName = '.\report\twonorm_v.txt';


fid=fopen(saveFileName,'w');
fclose(fid);

totalCycle = 5 ;                    % 5-fold cross validation
totalClass = size(dataSet , 2) ;    % total class of the dataset
dim = size(dataSet{1} , 2) ;        % dimension of the input samples



conf.kNum = 3 ;                     % Number of generated feature spaces
conf.C = 2^(-4);                    % parameter C
conf.Delta = [10^-1, 10^0, 10^1];   % parameters of the RBF kernel paramreters
conf.kType = {'g', 'g', 'g'} ;      % kernrl type
conf.kPar = kPar;                   % kernel parameter

cLen = 7;                           % 5-fold CV
finalRecord = [] ;
FinalRes = cell(cLen, 1) ;
count = 1;
segment = samples2Pieces(dataSet , totalCycle) ;
for cycle = 1 : cLen
    fprintf('******** C = %f *********\n' , 10^(cycle-4)) ;
    conf.C = 10^(cycle - 4) ;             % lamda
    
    res = [] ;
    forSave = cell(totalCycle, 1);
    for index_cycle = 1:totalCycle;
        testSet = [] ;
        for i = 1 : totalClass
            testSet = [testSet ; segment{index_cycle , i}] ;
        end
        trainSet = cell(1 , totalClass) ;
        for i = 1  : totalCycle
            if i ~= index_cycle
                for j = 1 : totalClass
                    trainSet{j} = [trainSet{j} ; segment{i , j}(:,1:dim-1)] ;
                end
            end
        end
        temp = NLMKSVM_MultiClass(trainSet , testSet , conf) ;
        regRate = temp.regRate;
        t_train = temp.t_train;
        forSave{index_cycle, 1} = temp;
        
        res = [res ; [regRate, t_train]] ;
        fid=fopen(saveFileName,'a');
        fprintf('The %d cycle--- Recog: %f;\n' , index_cycle , regRate) ;
        fprintf(fid,'The %d cycle--- Recog: %f;\n' , index_cycle , regRate) ;
        fclose(fid);
    end;
    res(totalCycle+1 , :) = mean(res) ;
    res(totalCycle+2 , :) = std(res(1:totalCycle , :)) ;
    FinalRes{count, 1} = res; FinalRes{count, 2} = forSave;
    count = count +1;
    fid=fopen(saveFileName,'a');
    fprintf('.......  Best Reg = %f\tstd = %f ........\t' , res(totalCycle+1 , 1) , res(totalCycle+2 , 1)) ;
    fprintf(fid,'.......  Std = %f\tstd = %f ........\t' , res(totalCycle+1 , 1) , res(totalCycle+2 , 1)) ;
    fprintf('....... Time = %f\ttstd = %f .......\n' , res(totalCycle+1 , 2) , res(totalCycle+2 , 2)) ;
    fprintf(fid,'....... Std Time = %f\ttstd = %f .......\n' , res(totalCycle+1 , 2) , res(totalCycle+2 , 2)) ;
    fclose(fid);
    
    finalRecord = [finalRecord ; [conf.C, res(totalCycle+1 , :), res(totalCycle+2 , :)]] ;
end
[maxValue , maxIndex] = max(finalRecord(:,2)) ;
maxRes = FinalRes{maxIndex, 1} ;
savedObj.maxRes = maxRes ;

fid=fopen(saveFileName,'a');
fprintf('.......  mean = %f\tstd = %f \t' , maxRes(totalCycle+1 , 1) , maxRes(totalCycle+2 , 1)) ;
fprintf(fid,'.......  mean = %f\tstd = %f \t' , maxRes(totalCycle+1 , 1) , maxRes(totalCycle+2 , 1)) ;
fprintf('....... meanTime = %f\ttstd = %f .......\n' , maxRes(totalCycle+1 , 2) , maxRes(totalCycle+2 , 2)) ;
fprintf(fid,'....... meanTime = %f\ttstd = %f .......\n' , maxRes(totalCycle+1 , 2) , maxRes(totalCycle+2 , 2)) ;
fclose(fid);

savedObj.FinalRes = FinalRes ;
savedObj.finalRecord = finalRecord ;
save(saveMatName , 'savedObj') ;
