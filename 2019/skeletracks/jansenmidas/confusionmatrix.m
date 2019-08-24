%%  CONFUSIONMATRIX.M
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

function [CFPixel,COMP] = confusionmatrix(IMG,GT)

%%% PRELIMINAR VARS AND PIXEL AMOUNT %%%
[M,N] = size(GT);
CFMatrix = zeros(M,N,4); CFPixel = zeros(1,4); %% FP, TP, FN, TN: 1st, 2nd, 3rd, 4th.
COMP = zeros(M,N,3);

for i = 1:M
    for j = 1:N
        if ((GT(i,j) == 0) && (IMG(i,j) ~= 0)) %% False Positive pixel
            CFMatrix(i,j,1) = 255;
            CFPixel(1) = CFPixel(1) +1;
        elseif ((GT(i,j) ~= 0) && (IMG(i,j) ~= 0)) %% True Positive pixel
            CFMatrix(i,j,2) = 255;
            CFPixel(2) = CFPixel(2) +1;
        elseif ((GT(i,j) ~= 0) && (IMG(i,j) == 0)) %% False Negative pixel
            CFMatrix(i,j,3) = 255;
            CFPixel(3) = CFPixel(3) +1;
        else %% True Negative pixel
            CFMatrix(i,j,4) = 255;
            CFPixel(4) = CFPixel(4) +1;
        end
    end
end

%%% GENERATING COLOR GT COMPARISON IMAGE %%%
for i = 1:3 %% red = FP; green = TP; blue = FN
    COMP(:,:,i) = CFMatrix(:,:,i);
end
