%%
%%  STARLET.M
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
%%         L, last desired decomposition level.
%%
%%  Output: S, starlet approximation levels.
%%          D, starlet detail levels.
%%
%%    * Other files required: main.m, binarize.m, confusionmatrix.m,
%% matthewscc.m, twodimfilt.m, xtracttracks.m
%%
%%    * Please cite:
%%
%% de Siqueira, A. F., Nakasuga, W. M., Pagamisse, A., Tello Saenz, C.
%% A., & Job, A. E. (2014). An automatic method for segmentation of
%% fission tracks in epidote crystal photomicrographs. Computers and
%% Geosciences, 69, 55–61. https://doi.org/10.1016/j.cageo.2014.04.008
%%


function [S,D] = starlet(IMG,L)

%%% PRELIMINAR VARS %%%
h = [1 4 6 4 1]; % 1D-h filter
IMG = double(IMG);
[M,N] = size(IMG); % image info
S = zeros(M,N,L); D = S; % resulting vectors: approximation and details

%%% MIRRORING CRITERIA: LOWER PIXEL NUMBER %%%
if M < N
    T = M;
else
    T = N;
end

auxS = padarray(IMG,[T T],'symmetric'); % mirroring

%%% STARLET APLICATION %%%
for i = 1:L
    prevIMG = auxS; % previous level image
    h2 = twodimfilt(h,i-1); % building 2D-h filter

    %%% WAVELET COEFFICIENTS %%%
    auxS = conv2(prevIMG,h2,'same'); % approximation coefficients
    auxD = prevIMG - auxS; % detail coefficients

    S(:,:,i) = auxS(t+1:m+t,t+1:n+t); % mirroring correction
    D(:,:,i) = auxD(t+1:m+t,t+1:n+t);
end
