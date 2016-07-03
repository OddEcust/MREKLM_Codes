function finalClass = svm_fuc(classOne, classTwo, test, conf, c1, c2)
    %
    % classOne, classTwo �������ţ� ѵ������
    % c1 Ϊ classOne ���ţ�c2 Ϊ classTwo ����
    % conf = {C, iter, kType, kPar}
    % test �������ţ� ��������
    %
    lenTest = length(test(:,1));
    finalClass = c1*ones(lenTest, 1);
    svm_res = svm_train_fuc(classOne, classTwo, conf);
    SVMStruct = svm_res.SVMStruct;
    res_test = svmclassify(SVMStruct.candidate,test);
    indX2 = find(res_test == -1) ;
    finalClass(indX2) = c2 ;
end