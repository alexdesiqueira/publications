%%
%%  MATTHEWSCC.M
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
%%  Input: TP, true positive pixels.
%%         TN, true negative pixels.
%%         FP, false positive pixels.
%%         FN, false negative pixels.
%%
%%  Output: MCC, Matthews correlation coefficient.
%%
%%    * Other files required: main.m, binarize.m, confusionmatrix.m,
%% starlet.m, twodimfilt.m, xtracttracks.m
%%
%%    * Please cite:
%%
%% de Siqueira, A. F., Nakasuga, W. M., Pagamisse, A., Tello Saenz, C.
%% A., & Job, A. E. (2014). An automatic method for segmentation of
%% fission tracks in epidote crystal photomicrographs. Computers and
%% Geosciences, 69, 55–61. https://doi.org/10.1016/j.cageo.2014.04.008
%%


function MCC = matthewscc(TP,TN,FP,FN)

MCC = ((TP*TN)-(FP*FN))/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
