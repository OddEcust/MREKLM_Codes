function Res = svm_OAO_fuc(train, test, conf)
    %
    % train = {c1, c2, c3,..., cm}, 最后一列为类标号
    % test = {testData, labels}
    % conf = {C, iter, kType, kPar}
    %
    
    totalClass = size(train, 2) ;
    [lenTest, dim] = size(test) ;
    testLabels = test(:,dim) ;
    test(:,dim) = [];
    TotalClass = zeros(lenTest, totalClass) ;
    tic;
    for c1 = 1 : totalClass
        classOne = train{c1};
        classOne(:,dim) = [];
        for c2 = c1+1 : totalClass
            classTwo = train{c2};
            classTwo(:,dim) = [];
            res_testLabel = svm_fuc(classOne, classTwo, test, conf, c1, c2);
            indX1 = find(res_testLabel == c1);
            indX2 = find(res_testLabel == c2);
            TotalClass(indX1, c1) = TotalClass(indX1, c1) +1;
            TotalClass(indX2, c2) = TotalClass(indX2, c2) +1;
        end 
    end
    t = toc;
    [maxVal, maxInd] = max(TotalClass');
    finalClass = maxInd';
    
    acc = length(find(finalClass == testLabels))/lenTest;
    Res.TotalClass = TotalClass;
    Res.fianlClass = finalClass;
    Res.acc = acc;   
    Res.t_train = t;
end