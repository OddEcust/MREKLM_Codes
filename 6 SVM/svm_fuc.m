function finalClass = svm_fuc(classOne, classTwo, test, conf, c1, c2)
    %
    % classOne, classTwo 不带类标号， 训练样本
    % c1 为 classOne 类标号，c2 为 classTwo 类标号
    % conf = {C, iter, kType, kPar}
    % test 不来类标号， 测试样本
    %
    lenTest = length(test(:,1));
    finalClass = c1*ones(lenTest, 1);
    svm_res = svm_train_fuc(classOne, classTwo, conf);
    SVMStruct = svm_res.SVMStruct;
    res_test = svmclassify(SVMStruct.candidate,test);
    indX2 = find(res_test == -1) ;
    finalClass(indX2) = c2 ;
end