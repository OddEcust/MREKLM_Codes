function svm_start(data, saveFileName, saveMatName, CMatrix, kType, kParMat)
    %
    % 5-fold CV
    % data = {c1, c2, ..., cn}, 最后一列为类标号
    % CMatrix 参数C 取值向量
    % kParMat 核参数kPar 取值向量
    % kType 核函数类型 ‘rbf’
    
    fid = fopen(saveFileName, 'w');
    fclose(fid); 
    
    CV_len = 5; 
    Segment = samples2Pieces(data, CV_len);
    totalClass = size(Segment,2);
    
    cLen = length(CMatrix);
    kParLen = length(kParMat);
    
    conf.kType = kType;
    conf.iter = 5000000;
    
    FinalRes = [];
    FinalSave = [] ;
    for ccycle = 1: cLen
        conf.C = CMatrix(ccycle);
        fid = fopen(saveFileName,'a');
        fprintf(fid,'====******===== C = %f ====******=====\n', conf.C);
        fprintf('====******===== C = %f ====******=====\n', conf.C);
        fclose(fid);
        
        for kParCycle = 1: kParLen
            conf.kPar = kParMat(kParCycle);
            ACC = [];
            tmp_Res = [] ;
            for CVcycle = 1: CV_len
                train = cell(1,totalClass);
                test = [];
                
                for classId = 1: totalClass
                    test = [test; Segment{CVcycle, classId}];
                end
                
                for i = 1 : CV_len
                    if i ~= CVcycle
                        for classId = 1: totalClass
                            train{classId} = [train{classId}; Segment{i,classId}];
                        end
                    end
                end
                Res = svm_OAO_fuc(train, test, conf);
                ACC = [ACC; Res.acc, Res.t_train];
                tmp_Res{CVcycle} = Res;
                
                fid = fopen(saveFileName,'a');
                fprintf(fid,'****\t CV %d --- ACC: %f \t****\n', CVcycle, Res.acc);
                fprintf('****\t CV %d --- ACC: %f \t****\n', CVcycle, Res.acc);
                fclose(fid);
            end
            ACC(CV_len+1: CV_len+2,:) = [mean(ACC,1); std(ACC,1)];
            FinalSave = [FinalSave; {conf.C, conf.kPar, tmp_Res, ACC}];
            FinalRes = [FinalRes; conf.C, conf.kPar, ACC(CV_len+1,1)];
            
            fid = fopen(saveFileName,'a');
            fprintf(fid,'************* Mean: %f(%f) \t Time: %f(%f) ***************\n', ACC(CV_len+1:CV_len+2,1)', ACC(CV_len+1:CV_len+2,2)');
            fprintf('************* Mean: %f(%f) \t Time: %f(%f) ***************\n', ACC(CV_len+1:CV_len+2,1)', ACC(CV_len+1:CV_len+2,2)');
            fclose(fid);
        end
    end
    
    [maxVal, maxId] = max(FinalRes(:,3));
    maxRes = FinalSave{maxId, 4};
    maxAcc = maxRes(CV_len+1:CV_len+2,1)';
    maxTime = maxRes(CV_len+1:CV_len+2,2)';
    
    fid = fopen(saveFileName,'a');
    fprintf(fid,'**---------- Best: %f(%f) \t Time: %f(%f) -------------**\n', maxAcc, maxTime);
    fprintf('**----------- Mean: %f(%f) \t Time: %f(%f) -------------**\n', maxAcc, maxTime);
    fclose(fid);
    
    SavedObj.FinalRes = FinalRes;
    SavedObj.FinalSave = FinalSave;
    SavedObj.maxAcc = maxAcc;
    SavedObj.maxRes = maxRes;
    SavedObj.maxTime = maxTime;
    save(saveMatName,'SavedObj');
end