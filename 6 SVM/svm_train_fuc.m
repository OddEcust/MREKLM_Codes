function Res = svm_train_fuc(classOne, classTwo, conf)
%
% classOne, classTwo ���� ���ţ�ÿһ��Ϊһ������
% conf = {C, iter, kType, kPar}
%
    [lenX1, dim] = size(classOne) ;
    lenX2 = size(classTwo,1);
    trainData = [classOne; classTwo] ;
    labels = [ones(lenX1,1); -ones(lenX2,1)];
    
    C = conf.C;
    iter = conf.iter;
    kType = conf.kType;
    kPar = conf.kPar;
    
    option=svmsmoset('MaxIter',iter);
    
    SVMStruct.candidate = svmtrain(trainData, labels,'kernel_function', kType,...
                        'rbf_sigma',kPar,'method','SMO','boxconstraint',C,'SMO_OPTS',option); %����ѵ��
    Res.SVMStruct = SVMStruct ;
    Res.C = C ;
    Res.kType = kType;
    Res.kPar = kPar;
    Res.iter = iter;
end