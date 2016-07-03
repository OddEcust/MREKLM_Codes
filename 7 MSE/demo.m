warning off ;
clear;
clc;

load('..\0 Datasets\twonorm_v.mat');
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

segment = samples2Pieces(dataSet , totalCycle) ;
FinalRes = [];
FinalSave = [];

saveRes = cell(totalCycle, 1);
res = [];
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
    [trnRes, t_train] = MSE_MultiClass(trainSet , testSet) ;
    saveRes{index_cycle} = trnRes;
    res = [res ; [trnRes.tstReg, t_train]] ;
    fid=fopen(saveFileName,'a');
    fprintf('The %d cycle--- Recog: %f;\n' , index_cycle , trnRes.tstReg) ;
    fprintf(fid,'The %d cycle--- Recog: %f;\n' , index_cycle , trnRes.tstReg) ;
    fclose(fid);
end;
res(totalCycle+1 , :) = mean(res) ;
res(totalCycle+2 , :) = std(res(1:totalCycle , :)) ;
FinalRes = [FinalRes; [res(totalCycle+1, :)]];
FinalSave = [FinalSave; {saveRes}, {res}];

fid=fopen(saveFileName,'a');
fprintf('.......  mean = %f\tstd = %f ........\t' , res(totalCycle+1 , 1) , res(totalCycle+2 , 1)) ;
fprintf(fid,'.......  mean = %f\tstd = %f ........\t' , res(totalCycle+1 , 1) , res(totalCycle+2 , 1)) ;
fprintf('....... meanTime = %f\ttstd = %f .......\n' , res(totalCycle+1 , 2) , res(totalCycle+2 , 2)) ;
fprintf(fid,'....... meanTime = %f\ttstd = %f .......\n' , res(totalCycle+1 , 2) , res(totalCycle+2 , 2)) ;
fclose(fid);




savedObj.FinalRes = FinalRes;
savedObj.FinalSave = FinalSave;

save(saveMatName, 'savedObj');

