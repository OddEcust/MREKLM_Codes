function [emp_train , emp_Test] = kernel_mapping(tmpTrn, train_data , test_data , kernelPerType , kernelPar)
    %
    % train_data = {train_class_one , train_class_two } ;
    % test_data = [X_in_input_space ] ;
    %
    
    
    [emp_train , emp_Test] = emp_Generator(tmpTrn, train_data , test_data , kernelPerType , kernelPar) ;    
end

function [emp_train, emp_Test] = emp_Generator(tmpTrn, trainData , testData , kType , kPar)
    % start clock for trainData
    implicitKernel = Kernel(tmpTrn , tmpTrn , kType , kPar) ;
    [pc , variances , explained] = pcacov(implicitKernel);

    i = 1 ;
    label = 0 ;
    while variances(i) >= 1e-3 ;
        if i+1 > size(variances,1) ;
            label = 1 ;
            break ;
        end;
        i = i + 1 ;    
    end;
    
    if label == 0 ;
        i = i - 1 ;
    end;

    index = 1 : i ;
    P = pc(: , index) ;
    R = diag(variances(index)) ;
    
    
    implicitKernel = Kernel(trainData , tmpTrn , kType , kPar) ;
    emp_train = implicitKernel * P * R^(-1/2) ;    
    kerTestMat = Kernel(testData ,tmpTrn , kType , kPar) ;
    emp_Test = kerTestMat * P * R^(-1/2) ;  
end

