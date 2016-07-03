warning off ;
clear;
clc;

load('..\0 Datasets\twonorm_v');
dataSet = twonorm_v.data;
kPar = twonorm_v.kPar;
clear twonorm_v;
saveMatName = '.\report\twonorm_v.mat';
saveFileName = '.\report\twonorm_v.txt';

CMatrix = [10^-2, 10^-1, 10^0, 10^1, 10^2, 10^3];
kType = 'rbf';
kParMat = [10^-1, 10^0, 10^1].* kPar;

svm_start(dataSet, saveFileName, saveMatName, CMatrix, kType, kParMat);
