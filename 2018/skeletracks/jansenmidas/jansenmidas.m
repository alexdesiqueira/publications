%%  JANSENMIDAS.M
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

function [D,R,COMP,MCC] = jansenmidas()

%%% INTRODUCTION %%%
disp('');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%% Welcome to Jansen-MIDAS %%%%%%%%%%%%%%%%');
disp('%%%%%%%%%% Microscopic Data Analysis Software %%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('');

%%% CHOOSING INITIAL AND LAST DETAIL LEVELS %%%
initL = input('Initial detail level to consider in segmentation: ');
if (isempty(initL) || ~isnumeric(initL))
    disp('Assuming default value, initial level equals 1. Continue...'); fflush(stdout);
    initL = 1;
end

L = input('Last detail level to consider in segmentation: ');
if (isempty(L) || ~isnumeric(L))
    disp('Assuming default value, last level equals 5. Continue...'); fflush(stdout);
    L = 5;
end

%%% OBTAINING ORIGINAL IMAGE %%%
IMGname = input('Please type the original image name: ','s'); 
IMG = imread(IMGname);

%%% CONVERTING RGB IMAGE TO GRAY %%%
if (length(size(IMG)) == 3)
    IMG = rgb2gray(IMG);
end

%%% APPLYING MLSS %%%
printf('Applying MLSS...\n'); fflush(stdout);
[D,R] = mlss(IMG,initL,L);

%%% APPLY MLSOS? %%%
GTapply = input('Do you want to apply MLSOS (uses GT image)? ','s'); 

if ((GTapply == 'Y') || (GTapply == 'y'))
    %%% OBTAINING GT AND ORIGINAL IMAGES %%%
    GTname = input('Please type a GT image name: ','s');
    GT = imread(GTname);

    %%% CONVERTING RGB IMAGE TO GRAY %%%
    if (length(size(GT)) == 3)
        IMG = rgb2gray(GT);
    end

    %%% APPLYING MLSOS %%%
    [COMP,MCC] = mlsos(R,GT,initL,L);
end

%%% SAVING OR SHOWING IMAGES %%%
SAV = input('Type Y to save images or any to show them: ','s');

if ((SAV == 'Y') || (SAV == 'y'))
    %%% SAVING D IMAGES %%%
    for i = 1:L
        printf('Saving detail image... Level: %d\n', i); fflush(stdout);
        imshow(D(:,:,i));
        print('-dtiff','-r300',strcat(IMGname,'-D',num2str(i),'.tif'));
    end

    %%% SAVING R IMAGES %%%
    for i = initL:L
        printf('Saving segmentation image... Level: %d\n', i); fflush(stdout);
        imshow(R(:,:,i));
        print('-dtiff','-r300',strcat(IMGname,'-R',num2str(i),'.tif'));
    end

    %%% SAVING COMP IMAGES %%%
    if ((GTapply == 'Y') || (GTapply == 'y'))
        for i = initL:L
            printf('Saving comparison image... Level: %d\n', i); fflush(stdout);
            imshow(COMP(:,:,:,i));
            print('-dtiff','-r300',strcat(IMGname,'-COMP',num2str(i),'.tif'));
        end
    end
else
    %%% SHOWING D IMAGES %%%
    for i = 1:L
        printf('Showing detail image... Level: %d\n', i); fflush(stdout);
        figure; imshow(D(:,:,i));
        title(strcat(IMGname,'-D',num2str(i)));
    end

    %%% SHOWING R IMAGES %%%
    for i = initL:L
        printf('Showing segmentation image... Level: %d\n', i); fflush(stdout);
        figure; imshow(R(:,:,i));
        title(strcat(IMGname,'-R',num2str(i)));
    end

    %%% SHOWING COMP IMAGES %%%
    if ((GTapply == 'Y') || (GTapply == 'y'))
        for i = initL:L
            printf('Showing comparison image... Level: %d\n', i); fflush(stdout);
            figure; imshow(COMP(:,:,:,i));
            title(strcat(IMGname,'-COMP',num2str(i)));
        end
    end
end

disp('End of processing. Thanks!');
