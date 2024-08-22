"""convert_label_1_channel.py
This file allows to create a 1 channel png file from the original data.
"""

import argparse
import json
import pandas as pd
import numpy as np
from PIL import Image
import os
import glob

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

def convert_single_image(path_input_file, config_json):

    path_input_file

    filename = os.path.basename(path_input_file)
    
    path_directory = config_json["output_directory"]
    
    filename_1_channel = f"{filename[:-4]}_1ch.png"
    
    path_directory_new_file = os.path.join(path_directory, filename_1_channel)
    
    img_grayscale = Image.open(path_input_file).convert('L')
    
    img_grayscale.save(path_directory_new_file)
    
    print("Hello World")

def convert_images(config_json):
    
    data_directory = config_json["data_directory"]

    if data_directory[-1] == "/":
        data_directory = data_directory[:-1]

    l_path_labeled_files = glob.glob(f'{data_directory}/*_labeled.png', recursive=True)
    
    for path_file in l_path_labeled_files:
        convert_single_image(path_file, config_json)

def main():
    args = get_parser()
    config_json = get_json(args)
    
    # path_input_file = "/home/pcallec/nnUNet_blood-vessel/data/5000_All-Doppler-Labeled_Results/0001_labeled.png"

    # convert_single_image(path_input_file, config_json)
    
    convert_images(config_json)
    
if __name__ == "__main__":
    main()