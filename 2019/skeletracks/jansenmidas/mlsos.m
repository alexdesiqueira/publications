%%  MLSOS.M
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

function [COMP,MCC] = mlsos(R,IMGGT,initL,L)

%%% PREALLOCATING VARS %%%
[M,N] = size(IMGGT); % info
COMP = zeros(M,N,3,L); % COMP is a set of RGB images

for i = initL:L
    %%% COMPARISON PROGRAM %%%
    [auxPX,auxCOMP] = confusionmatrix(R(:,:,i),IMGGT);
    COMP(:,:,:,i) = auxCOMP; %COMP is a set of RGB images

    %%% CALCULATING MCC %%%
    MCC(i) = matthewscc(auxPX(1),auxPX(2),auxPX(3),auxPX(4));
end

%%% OPTIMAL SEGMENTATION LEVEL: HIGHER MCC %%%
figure; bar(MCC*100,'facecolor','r','edgecolor','r'); %% presenting MCC as percentage
title('MCC for each level'); xlabel('Level'); ylabel('Result');

end
