%%  MLSSORIGAUX.M
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

function R = mlssorigaux(IMG,D,initL)

[M,N,L] = size(D); % info
dMAX = zeros(M,N,L);
sum = 0;

for i = initL:L
    ratio = max(max(D(:,:,i))); ratio = 255/ratio; %% normalization
    dMAX(:,:,i) = uint8(ratio*D(:,:,i));

    sum = sum + dMAX(:,:,i);
end

R = uint8(sum) - uint8(IMG);
