%%  MLSS.M
%%
%%  Version: november 2014.
%%
%%  This file is part of the supplementary material to 'An automatic 
%% method for segmentation of fission tracks in epidote crystal 
%% photomicrographs, based on starlet wavelets'.
%%
%%  Author: 
%% Alexandre Fioravante de Siqueira, siqueiraaf@gmail.com
%%
%%  Description: this software (...)
%%
%%
%%
%%  Input: (...)
%%         (...)
%%
%%  Output: (...)
%%          (...)
%%          
%%  Other files required: (...)
%%
%%  Please cite:
%% (...)
%%

function [D,R] = mlss(IMG,initL,L,ALGchoice)

%%% PREALLOCATING VARS %%%
[M,N] = size(IMG); % info
R = zeros(M,N,L);

%%% STARLET APPLICATION %%%
[~,D] = starlet(IMG,L);

%%% CHOOSING MLSS ALGORITHM %%%
%ALGchoice = input('Type V for Variant or any to Original MLSS: ','s');

if ((ALGchoice == 'v') || (ALGchoice == 'V')) %% apply Variant algorithm
    for i = initL:L
        aux = mlssvaraux(IMG,D(:,:,[1:i]),initL); %% segmentation
        R(:,:,i) = ~binarize(aux); %% binarize input image
    end
else %% apply Original algorithm
    for i = initL:L
        aux = mlssorigaux(IMG,D(:,:,[1:i]),initL); %% segmentation
        R(:,:,i) = binarize(aux); %% binarize input image
    end
end
