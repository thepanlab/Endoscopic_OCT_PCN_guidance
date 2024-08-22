"""structure_data.py
This code will stucture in the following way:

data/
├── training/
│   ├── input/
│   │   ├── img1
│   │   ├── img2
│   │   ├── img3
│   │       (...)
│   └── output/
│       ├── img1
│       ├── img2
│       ├── img3
│           (...)
└── testing/
    ├── input/
    │   ├── img1
    │   ├── img2
    │   ├── img3
    │       (...)
    └── output/
        ├── img1
        ├── img2
        ├── img3
            (...)
            
Assuming that the input data is:

data/
├── 0001.png
├── 0001_labeled_1ch.png
├── 0002.png
├── 0002_labeled_1ch.png
├── 0003.png
├── 0003_labeled_1ch.png
(...)
"""

import argparse
import json
from multiprocessing.sharedctypes import Value
import pandas as pd
import os
import numpy as np
import re
from collections import Counter  
from PIL import Image
import os
import glob
from sklearn.model_selection import train_test_split

from typing import Tuple, List, Union
from skimage import io
import SimpleITK as sitk
import numpy as np
import tifffile

# Function taken from /nnUNet/nnunet/utilities/file_conversions.py to debug
def convert_2d_image_to_nifti(input_filename: str, output_filename_truncated: str, spacing=(999, 1, 1),
                              transform=None, is_seg: bool = False) -> None:
    """
    Reads an image (must be a format that it recognized by skimage.io.imread) and converts it into a series of niftis.
    The image can have an arbitrary number of input channels which will be exported separately (_0000.nii.gz,
    _0001.nii.gz, etc for images and only .nii.gz for seg).
    Spacing can be ignored most of the time.
    !!!2D images are often natural images which do not have a voxel spacing that could be used for resampling. These images
    must be resampled by you prior to converting them to nifti!!!

    Datasets converted with this utility can only be used with the 2d U-Net configuration of nnU-Net

    If Transform is not None it will be applied to the image after loading.

    Segmentations will be converted to np.uint32!

    :param is_seg:
    :param transform:
    :param input_filename:
    :param output_filename_truncated: do not use a file ending for this one! Example: output_name='./converted/image1'. This
    function will add the suffix (_0000) and file ending (.nii.gz) for you.
    :param spacing:
    :return:
    """
    img = io.imread(input_filename)

    if transform is not None:
        img = transform(img)

    if len(img.shape) == 2:  # 2d image with no color channels
        img = img[None, None]  # add dimensions
    else:
        assert len(img.shape) == 3, "image should be 3d with color channel last but has shape %s" % str(img.shape)
        # we assume that the color channel is the last dimension. Transpose it to be in first
        img = img.transpose((2, 0, 1))
        # add third dimension
        img = img[:, None]

    # image is now (c, x, x, z) where x=1 since it's 2d
    if is_seg:
        assert img.shape[0] == 1, 'segmentations can only have one color channel, not sure what happened here'

    for j, i in enumerate(img):

        if is_seg:
            i = i.astype(np.uint32)

        itk_img = sitk.GetImageFromArray(i)
        itk_img.SetSpacing(list(spacing)[::-1])
        if not is_seg:
            sitk.WriteImage(itk_img, output_filename_truncated + "_%04.0d.nii.gz" % j)
        else:
            sitk.WriteImage(itk_img, output_filename_truncated + ".nii.gz")

def get_parser():
    """
    Obtains arguments parser

    Arguments:
        None
    Returns:
        ArgumentParset args
    """
    # https://www.loekvandenouweland.com/content/using-json-config-files-in-python.html
    parser = argparse.ArgumentParser()

    parser.add_argument("-j","--load_json", required = True,
            help='Load settings from file in json format.')

    args = parser.parse_args()

    return args
# print(parser)

def get_json(args):
    """
    Obtains configurations stores in json file

    Arguments:
        ArgumentParser: args
    Returns:
        dict config_json: it contains all the parameters from json file
    """
    with open(args.load_json) as config_file:
        config_json = json.load(config_file)

    return config_json

def get_number_from_filename(filename):
    return int(filename[:4])

def get_ordered_list_files_and_labels(config_json):
    # lets differentiate by length because it is fixed
    data_directory = config_json["data_path"]
    
    # get list files
    l_path_files = glob.glob(f'{data_directory}/????.png', recursive=True)
    
    # get list of labels
    l_path_labeled_files = glob.glob(f'{data_directory}/*_labeled_1ch.png', recursive=True)
       
    # to do 
    # sort and return
    l_path_files.sort()
    l_path_labeled_files.sort()
    
    return l_path_files, l_path_labeled_files

def split_data(config_json):
    
    l_path_files, l_path_labeled_files = get_ordered_list_files_and_labels(config_json)
    
    l_train_test_percentage = config_json["division_percentage_train_test"]

    l_path_file_train, l_path_file__test, l_path_label_train, l_path_label_test = train_test_split(l_path_files,
                                                                                           l_path_labeled_files,
                                                                                        test_size = l_train_test_percentage[1]/100, random_state=1234)

    return l_path_file_train, l_path_file__test, l_path_label_train, l_path_label_test

def get_data_new_structure(config_json):
    l_path_file_train, l_path_file_test, l_path_label_train, l_path_label_test = split_data(config_json)

    path_output_directory = config_json["output_path"]

    path_training_input = os.path.join(path_output_directory, "training","input")
    path_training_output = os.path.join(path_output_directory, "training","output")
    path_testing_input = os.path.join(path_output_directory, "testing","input")
    path_testing_output = os.path.join(path_output_directory, "testing","output")

    os.makedirs(path_training_input, exist_ok = True)
    os.makedirs(path_training_output, exist_ok = True)
    os.makedirs(path_testing_input, exist_ok = True)
    os.makedirs(path_testing_output, exist_ok = True)
    
    # training
    for training_path in l_path_file_train:
        # copy from original path to path_training_input
        os.system(f"cp {training_path} {path_training_input}")

    for i, training_path in enumerate(l_path_label_train):
        # copy from original path to path_training_input directory
        os.system(f"cp {training_path} {path_training_output}")

        # Change name of copied file
        path_original_name = os.path.join(path_training_output, os.path.basename(training_path))

        filename_image = os.path.basename(l_path_file_train[i])
        path_new_name = os.path.join(path_training_output, filename_image)
        os.system(f"mv {path_original_name} {path_new_name}")

    for testing_path in l_path_file_test:
        # copy from original path to path_training_input
        os.system(f"cp {testing_path} {path_testing_input}")

    for i, testing_path in enumerate(l_path_label_test):
        # copy from original path to path_training_input directory
        os.system(f"cp {testing_path} {path_testing_output}")
        # Change name of copied file
        path_original_name = os.path.join(path_testing_output, os.path.basename(testing_path))

        filename_image = os.path.basename(l_path_file_test[i])
        path_new_name = os.path.join(path_testing_output, filename_image)
        os.system(f"mv {path_original_name} {path_new_name}")

    
def verify_label_ok(path, n_segments):
    # Convert image to grayscale
    b_return = False

    img_grayscale = Image.open().convert('L')

    c_temp = Counter(a_img_road.ravel())
    if len(c_temp.keys()) == n_segments:
        b_return = True
    
    return b_return

def main():
    args = get_parser()
    config_json = get_json(args)
    
    # one_path = "/home/pcallec/nnUNet_blood-vessel/data/5000_All-Doppler-Labeled_Results/0001_labeled.png"
    # output_path = "/home/pcallec/nnUNet_blood-vessel/results/5000_All-Doppler-Labeled_Results/0001_labeled.png"
    # convert_2d_image_to_nifti(one_path, output_path, is_seg=True)
    # verify_label_ok(one_path, n_segments = 2)
    # filename = "0256.png"
    # number = get_number_from_filename(filename)

    # get_ordered_list_files_and_labels(config_json)

    get_data_new_structure(config_json)

if __name__ == "__main__":
    main()