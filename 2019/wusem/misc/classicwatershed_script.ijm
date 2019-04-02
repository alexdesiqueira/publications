function process_image(input, output, filename) {
    open(input + filename);
    setOption("BlackBackground", true);
    run("Make Binary");
    getDimensions(width, height, dummy, dummy, dummy);

    run("Fill Holes");
    run("Watershed");

    run("Canvas Size...", "width=" + width+2 + " height=" + height+2 + " position=Bottom-Right zero");
    run("Kill Borders");
    run("Canvas Size...", "width=" + width-2 + " height=" + height-2 + " position=Bottom-Right zero");

    run("Analyze Particles...", "size=64-Infinity show=Outlines display clear");
    saveAs("Results", output + filename + "-results.txt");

    // please uncomment the following line to generate the resulting images.
    saveAs("Tiff", output + filename + "-outlines.tif");

    run("Close All");
}

function process_folder(input) {
    list = getFileList(input);
    for (i = 0; i < list.length; i++) {
        if (endsWith(list[i], "/")) {
                process_folder("" + input + list[i]);
        }
        else if (endsWith(list[i], extension))  {
                process_image(input, output, list[i]);
        }
    }
}

// Helping variable.

output = "../res_figures/fig_watershed";

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
