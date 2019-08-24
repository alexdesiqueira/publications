function process_image(input, output, seed, filename) {
    // first part: getting threshold value to H-watershed.
    open(input + filename);
    if (!is("grayscale")) {
        run("8-bit");
    }

    setOption("BlackBackground", true);
    setAutoThreshold("IsoData white");

    getThreshold(dummy, upper);
    close();

    // second part: applying H-watershed.
    open(input + filename);
    main_img = getImageID();
    if (!is("grayscale")) {
        run("8-bit");
    }

    setOption("BlackBackground", true);
    getDimensions(width, height, dummy, dummy, dummy);
    run("Invert");

    // setting seed and flooding.
    peak_flooding = 100;

    run("H_Watershed", "impin=["+getTitle()+"] hmin="+seed+" thresh="+upper+" peakflooding="+peak_flooding+" outputmask=true allowsplitting=false");
    run("Canvas Size...", "width=" + width+2 + " height=" + height+2 + " position=Bottom-Right zero");
    run("Kill Borders");
    run("Canvas Size...", "width=" + width-2 + " height=" + height-2 + " position=Bottom-Right zero");

    run("Analyze Particles...", "size=64-Infinity show=Outlines display clear");
    saveAs("Results", output + filename + "_seed_" + toString(seed) + "-results.txt");

    // please uncomment the following line to generate the resulting images.
    // saveAs("Tiff", output + filename + "_seed_" + toString(seed) + "-outlines.tif");

    selectImage(main_img);
    close("\\Others");

    run("Close All");
}

function process_folder(input, seed) {
    list = getFileList(input);
    for (i = 0; i < list.length; i++) {
        if (endsWith(list[i], "/")) {
                process_folder("" + input + list[i], seed);
        }
        else if (endsWith(list[i], extension)) {
                process_image(input, output, seed, list[i]);
        }
    }
}

base_folder = "/home/alex/documents/publications/2019/wusem/misc/"

// Helping variable.

output = base_folder + "../figures/res_figures/fig_hwatershed/";

// Directory Kr-78_4,5min, from dataset_01.
seed = 5;
extension = ".bmp"
input = base_folder + "../figures/orig_figures/dataset_01/Kr-78_4,5min/";
process_folder(input, seed);

// Directory Kr-78_8,5min, from dataset_01.
seed = 5;
extension = ".bmp"
input = base_folder + "../figures/orig_figures/dataset_01/Kr-78_8,5min/";
process_folder(input, seed);

// dataset_02.
seed = 19
extension = ".jpg"
input = base_folder + "../figures/orig_figures/dataset_02/*.MAG1.jpg";
process_folder(input, seed);

seed = 23
extension = ".jpg"
input = base_folder + "../figures/orig_figures/dataset_02/*.MAG2.jpg";
process_folder(input, seed);
