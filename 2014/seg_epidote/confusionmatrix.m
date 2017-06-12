%%
%%  CONFUSIONMATRIX.M
%%
%% Copyright (C) 2014 Alexandre Fioravante de Siqueira
%%
%% This file is part of the supplementary material to 'An automatic
%% method for segmentation of fission tracks in epidote crystal
%% photomicrographs, based on starlet wavelets'.
%%
%% This program is distributed in the hope that it will be useful,
%% but WITHOUT ANY WARRANTY; without even the implied warranty of
%% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%% GNU General Public License for more details.
%%
%% You should have received a copy of the GNU General Public License
%% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%%
%%    * Authors:
%% Alexandre Fioravante de Siqueira, siqueiraaf@gmail.com
%% Wagner Massayuki Nakasuga, wamassa@gmail.com
%% Aylton Pagamisse, aylton@fct.unesp.br
%% Carlos Alberto Tello Saenz, tello@fct.unesp.br
%% Aldo Eloizo Job, job@fct.unesp.br
%%
%%    * Description: this software applies algorithms to segment
%% fission-tracks in crystal images by optical microscopy, based on
%% starlets. Automatization of these algorithms is given using Matthews
%% Correlation Coefficient (MCC). The difference between an image and
%% its Ground Truth is given by a colored comparison.
%%
%%  Input: IMG, a gray input image.
%%         IMGGT, a ground truth image obtained from IMG.
%%
%%  Output: TPpxl, amount of true positive pixels.
%%          TNpxl, amount of true negative pixels.
%%          FPpxl, amount of false positive pixels.
%%          FNpxl, amount of false negative pixels.
%%          COMP, colored comparison between IMG and IMGGT.
%%
%%    * Other files required: main.m, binarize.m, mattewscc.m, starlet.m,
%% twodimfilt.m, xtracttracks.m
%%
%%    * Please cite:
%%
%% de Siqueira, A. F., Nakasuga, W. M., Pagamisse, A., Tello Saenz, C.
%% A., & Job, A. E. (2014). An automatic method for segmentation of
%% fission tracks in epidote crystal photomicrographs. Computers and
%% Geosciences, 69, 55–61. https://doi.org/10.1016/j.cageo.2014.04.008
%%


function [TPpxl,TNpxl,FPpxl,FNpxl,COMP] = confusionmatrix(IMGGT,IMG)

%%% PRELIMINAR VARS %%%
aux = size(IMGGT);
K = aux(1);
L = aux(2);
comp = zeros(K,L,3);

TP = TN = FP = FN = aux; %% initializing with same size
TPpxl = TNpxl = FPpxl = FNpxl = 0; %% initialize sums

%%% COMPARISON BETWEEN IMG AND IMGGT %%%
for i = 1:1:K
    for j = 1:1:L
        if (IMGGT(i,j) != 0 && IMG(i,j) != 0) %% True Positive
            TP(i,j) = 255;
            TN(i,j) = FP(i,j) = FN(i,j) = 0;
            TPpxl = TPpxl +1;

        elseif (IMGGT(i,j) == 0 && IMG(i,j) != 0) %% False Positive
            FP(i,j) = 255;
            TP(i,j) = TN(i,j) = FN(i,j) = 0;
            FPpxl = FPpxl +1;

        elseif (IMGGT(i,j) != 0 && IMG(i,j) == 0) %% False Negative
            FN(i,j) = 255;
            TP(i,j) = TN(i,j) = FP(i,j) = 0;
            FNpxl = FNpxl +1;

        else    %% True Negative
            TP(i,j) = FP(i,j) = FN(i,j) = 0;
            TNpxl = TNpxl +1;
        end
    end
end

prec = (TPpxl / (TPpxl + FPpxl)) * 100
rec = (TPpxl / (TPpxl + FNpxl)) * 100
spec = (TNpxl / (TNpxl + FPpxl)) * 100
accur = ((TPpxl + TNpxl) / (TPpxl + TNpxl + FNpxl + FPpxl)) * 100

%%% GENERATING COLOR COMPARISON %%%
COMP(:,:,1) = FP; %% red   = False Positive
COMP(:,:,2) = TP; %% green = True Positive
COMP(:,:,3) = FN; %% blue  = False Negative
