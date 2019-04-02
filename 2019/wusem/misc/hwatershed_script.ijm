function process_image(input, output, filename) {
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

    for (seed = 1; seed < 61; seed++) {
        run("H_Watershed", "impin=["+getTitle()+"] hmin="+seed+" thresh="+upper+" peakflooding="+peak_flooding+" outputmask=true allowsplitting=false");
        run("Canvas Size...", "width=" + width+2 + " height=" + height+2 + " position=Bottom-Right zero");
        run("Kill Borders");
        run("Canvas Size...", "width=" + width-2 + " height=" + height-2 + " position=Bottom-Right zero");

        run("Analyze Particles...", "size=64-Infinity show=Outlines display clear");
        saveAs("Results", output + "/results/" + filename + "_seed_" + toString(seed) + "-results.txt");

        // please uncomment the following line to generate the resulting images.
        // saveAs("Tiff", output + "/figures/" + filename + "_seed_" + toString(seed) + "-outlines.tif");

        selectImage(main_img);
        close("\\Others");
    }
    run("Close All");
}

function process_folder(input) {
    list = getFileList(input);
    for (i = 0; i < list.length; i++) {
        if (endsWith(list[i], "/")) {
                process_folder("" + input + list[i]);
        }
        else if (endsWith(list[i], extension)) {
                process_image(input, output, list[i]);
        }
    }
}

// Helping variable.

output = "../res_figures/fig_hwatershed/";

// Directory Kr-78_4,5min, from dataset_01.

extension = ".bmp"
input = "../orig_figures/dataset_01/Kr-78_4,5min/";
process_folder(input);

// Directory Kr-78_8,5min, from dataset_01.

extension = ".bmp"
input = "../orig_figures/dataset_01/Kr-78_8,5min/";
process_folder(input);

// dataset_02.

extension = ".jpg"
input = "../orig_figures/dataset_02/";
process_folder(input);
