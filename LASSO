clear all;close all;clc 

% The dictionary learning follows the K-SVD algorithm, 
% and uses l1 framework from SPGL1 toolbox. Please refer to 
% K-SVD toolbox: http://www.cs.technion.ac.il/~elad/software/
% SPGL1 toolbox: http://www.cs.ubc.ca/~mpf/spgl1/citing.html
% for completed codes and licence information.


% Please refer to the following paper for the details:
% Ding, Yacong, and Bhaskar D. Rao. "Dictionary Learning Based Sparse Channel Representation and Estimation for FDD Massive MIMO Systems." 
% arXiv preprint arXiv:1612.06553 (2016).

% Yasser 



load('ChannelTraining.mat') % load channel response for dicitionary learning
[N, L] = size(H);
M = 400;  % column number of the dictionary matrix
H_d=H;
    
%% genearte DFT dictionary and DFT basis
t=0:N-1;
alpha=0.5;
theta=(-1:2/M:1-2/M)*alpha;
Dic_DFT_M=1/sqrt(N)*exp(1j*2*pi*t.'*theta);
Dic_DFT_M = Dic_DFT_M./(ones(N,1)*sqrt(sum(abs(Dic_DFT_M).^2)));

t=0:N-1;
alpha=0.5;
theta=(-1:2/N:1-2/N)*alpha;
Dic_DFT_N=1/sqrt(N)*exp(1j*2*pi*t.'*theta);
Dic_DFT_N = Dic_DFT_N./(ones(N,1)*sqrt(sum(abs(Dic_DFT_N).^2)));

%% dictionary learning
% paratmeter setting 
param.K = M;                                    % number of columns in the dictionary 
param.numIteration = 50;                        % number of iteration to execute the K-SVD algorithm.
param.preserveDCAtom = 0;                       
param.InitializationMethod =  'GivenMatrix';    
param.initialDictionary = Dic_DFT_M;            % use overcomplete DFT dictionary as starting point
param.displayProgress = 1;
Sigma=[0.1 0.01];
Sigma = 0.1;                                    

sparsity=zeros(1,length(Sigma));                % average number of nonzero elements
NNZ=zeros(length(Sigma),L);                     % histgram of nonzero elements
MSE_DIC=zeros(1,length(Sigma));                 % mean square error 

for wd=1:length(Sigma)

    % parameter setting
    sigma=Sigma(wd);
    if sigma<1
        param.errorGoal = sigma;
        param.errorFlag=1;
    else
        param.L=sigma;
        param.errorFlag=0;
    end
    param_d=param;

    % dictionary learning
    [Dictionary,output]  = KSVD(H_d,param_d);
    sparsity(wd)=(output.numCoef(end));
    NNZ(wd,:)=sum(output.CoefMatrix~=0);
    MSE_DIC(wd)=norm(H_d-Dictionary*output.CoefMatrix,'fro')^2/L;
    save([num2str(wd) 'LearnedDictionary.mat'],'Dictionary')
end
    
 
    
    
    
