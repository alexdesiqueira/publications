
# adapted from:
# https://stackoverflow.com/questions/22287375/sorting-filenames-in-numerical-order-in-python

import glob
import os
import re


def try_integer(string):
    try:
        return int(string)
    except ValueError:
        return string


def alphanum_key(string):
    return [try_integer(char) for char in re.split('([0-9]+)', string)]


def sort_nicely(list):
    return sorted(list, key=alphanum_key)


def read_water_tracks(path='../res_figures/fig_watershed/dataset_01/Kr-78_4,5min'):
    '''
    read_tracks('dataset_01/Kr-78_4,5min')
    '''

    # preparing the text file to receive track countings, according
    # to the dataset.
    count_path = '../counting/water_count/'
    if 'dataset_01' in path:
        folder = path.split(sep='/')[-1]
        file = open((count_path + 'water_dataset01_' +
                     folder + '_incid.txt'), 'w')
        file.write(('folder,image,auto_count\n'))
    elif 'dataset_02' in path:
        file = open(count_path + 'water_dataset02.txt', 'w')
        file.write(('image,auto_count\n'))

    for root, folders, files in os.walk(path):
        folders.sort()
        for fil in sort_nicely(files):
            if fil.endswith('.txt'):
                with open(root + '/' + fil) as f:
                    lines = f.readlines()
                    count = lines[-1].split()[0]  # number of tracks

                    if 'dataset_01' in path:
                        # order: folder, image, number of tracks.
                        line_file = (root.split(sep='/')[-1] + ',' +
                                     fil.split(sep='.')[0].split(sep='_')[-1] +
                                     ',' + count + '\n')
                    elif 'dataset_02' in path:
                        # order: image, number of tracks.
                        line_file = (fil.split(sep='.')[2] + ',' +
                                     count + '\n')
                    file.write(line_file)

    file.close()

    return None


def read_hwater_tracks(path='../res_figures/fig_hwatershed/dataset_01/Kr-78_4,5min'):
    '''
    read_tracks('dataset_01/Kr-78_4,5min')
    '''

    # preparing the text file to receive track countings, according
    # to the dataset.
    count_path = '../counting/hwater_count/'
    if 'dataset_01' in path:
        folder = path.split(sep='/')[-1]
        file = open((count_path + 'hwater_dataset01_' +
                     folder + '_incid.txt'), 'w')
        file.write(('folder,image,seed,auto_count\n'))
    elif 'dataset_02' in path:
        file = open(count_path + 'hwater_dataset02.txt', 'w')
        file.write(('image,seed,auto_count\n'))

    for root, folders, files in os.walk(path):
        folders.sort()
        for fil in sort_nicely(files):
            if fil.endswith('.txt'):
                with open(root + '/' + fil) as f:
                    lines = f.readlines()
                    count = lines[-1].split()[0]  # number of tracks

                    if 'dataset_01' in path:
                        # order: folder, image, number of tracks.
                        line_file = (root.split(sep='/')[-1] + ',' +  # folder name
                                     fil.split(sep='.')[0].split(sep='_')[-1] +  # number of image
                                     ',' + fil.split(sep='_')[-1].split('-')[0] +  # seed
                                     ',' + count + '\n')  # number of tracks
                    elif 'dataset_02' in path:
                        # order: image, number of tracks.
                        line_file = (fil.split(sep='.')[2] + ',' +  # number of image
                                     fil.split(sep='_')[-1].split('-')[0] +  # seed
                                     ',' + count + '\n')  # number of tracks
                    file.write(line_file)

    file.close()

    return None


def store_dump_files(path='../res_figures/fig_watershed/dump'):
    """
    """

    minutes = ['4,5min', '8,5min']
    samples = ['K0_incid', 'K20_incid', 'K30_incid', 'K40_incid',
               'K50_incid', 'K60_incid', 'K70_incid', 'K80_incid',
               'K90_incid']

    dataset01_folder = './hwater_figures/dataset_01/'
    dataset02_folder = './hwater_figures/dataset_02/'

    try:
        if not os.path.exists(dataset01_folder):
            os.makedirs(dataset01_folder)
        if not os.path.exists(dataset02_folder):
            os.makedirs(dataset02_folder)
    except:
        raise

    for _, _, files in os.walk(path):
        for fil in files:
            if 'FT-Lab' in fil:  # belongs to dataset_02
                new_file = dataset02_folder + fil
                old_file = path + '/' + fil
                os.rename(old_file, new_file)

            else:  # belongs to dataset_01
                for minute in minutes:
                    if minute in fil:
                        new_folder = dataset01_folder + 'Kr-78_' + \
                                     minute +'/'
                        try:
                            if not os.path.exists(new_folder):
                                os.makedirs(new_folder)
                        except:
                            raise

                        for sample in samples:
                            new_folder = dataset01_folder + 'Kr-78_' + \
                                        minute +'/' + sample + '/'
                            try:
                                if not os.path.exists(new_folder):
                                    os.makedirs(new_folder)
                            except:
                                raise

                            if sample in fil:
                                new_file = new_folder + fil
                                old_file = path + '/' + fil
                                os.rename(old_file, new_file)
    return None

"""
for root, folders, files in os.walk(path):
    for folder in sort_nicely(folders):
        if folder.endswith('incid'):
            print(root + '/' + folder)
            files = sort_nicely(os.listdir(root + '/' + folder))
            for f in files:
                if f.endswith('.txt'):
                    print(f)
"""
