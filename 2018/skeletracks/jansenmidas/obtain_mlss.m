pkg load image signal control

filenames = glob("../filt_figures/*.png");

for i = 1:numel(filenames)
    aux = imread(filenames{i});
    if(ndims(aux) > 2)
        aux = rgb2gray(aux);
    end
    [d, r] = mlss(aux, 2, 7, 'V');

    % building the binary MLSS filename.
    bin_fname = strcat("../auto_count/mlss/",
                       strsplit(filenames{i}, value="/"){3}(1:end-4),
                       ".csv");

    % writing the csv image to disk.
    csvwrite(bin_fname, r(:, :, 6));
end
