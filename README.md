# Segmentation

We used  [nnUnet v1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1), which at that moment was the latest one. Currently, there is an improved version [nnUnet v2](https://github.com/MIC-DKFZ/nnUNet/tree/master) 
## Installation nnUNet

Install nnUNet from [link](https://github.com/MIC-DKFZ/nnUNet). Install it as integrative framework (-e) because we will need to modify some files later:

```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```
## Data conversion

We need to comply with the requirements of the data provided in [link](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md)

<ins>Important notes</ins>:
* Please make your custom task ids start at 500 to ensure that there will be no conflicts with downloaded pretrained models!!! (IDs cannot exceed 999)

* The image files can have any scalar pixel type. The label files must contain segmentation maps that contain consecutive integers, starting with 0: 0, 1, 2, 3, ... num_labels. 0 is considered background.

`Task120_Massachusetts_RoadSegm.py` example [(link)](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task120_Massachusetts_RoadSegm.py) is used as a template to convert the data from 2d to pseudo-3d.

<ins>Note</ins>: `Task120_Massachusetts_RoadSegm.py` deals the situation when the output(labels) are only available for some of the images by doing a join (Line 63/64)

**Procedure**

In order to use the `Task120_Massachusetts_RoadSegm.py`. First, change the structure of your data to and convert labelled image to one channel:

```
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
```

<ins>Note</ins>: the label and image names should be the same in `input/` and `output/` directories.

*`convert_label_1_channel.py`: it converts the label images to 1 channel

*`structure_data.py`: it convert the data to the structure necesesary to be used by `Task120_Massachusetts_RoadSegm.py`

```bash
python convert_label_1_channel.py -j convert_label_1_channel_5000_v3.json

python stucture_data.py -j structure_data_5000_v3.json
```


`Task500_BloodVessel.py` and `Task501_BloodVessel_extra_test.py` will make the tranformation from png to nifti files to data. Thse files are based on `Task120_Massachusetts_RoadSegm.py`.

These filese can be found in `/scripts/conversion_to_nifti`. 

In order to be able to run `Task500_BloodVessel.py`, the following variables need to be defined, according to [link](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md).

```
export nnUNet_raw_data_base="/home/pcallec/nnUNet_blood-vessel/results/nnUNet_raw_data_base"
export nnUNet_preprocessed="/home/pcallec/nnUNet_blood-vessel/results/nnUNet_preprocessed"
export RESULTS_FOLDER="/home/pcallec/nnUNet_blood-vessel/results/nnUNet_trained_models"
```

# Experiment planning and preprocessing

`
nnUNet_plan_and_preprocess -t 500 --verify_dataset_integrity -pl3d None
`

# Run training

`
nnUNet_train 2d nnUNetTrainerV2 500 FOLD
`

Scripts can be seen in: `/scripts/scripts_nnunet/commands_all.sh`

**Note**: there was error `nnU-Net training: Error: mmap length is greater than file size and EOFError`. According to (Link)[https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/common_problems_and_solutions.md#nnu-net-training-error-mmap-length-is-greater-than-file-size-and-eoferror], the files .npy should be deleted.

# Run inference on test data
From documentation:
```
Note that per default, inference will be done with all available folds. We very strongly recommend you use all 5 folds. Thus, all 5 folds must have been trained prior to running inference. The list of available folds nnU-Net found will be printed at the start of the inference.
```

Therefore prediction is using 5 folds:

```bash
nnUNet_find_best_configuration -m 2d -t 500

CUDA_VISIBLE_DEVICES=1 nnUNet_predict -i /home/pcallec/nnUNet_blood-vessel/results/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BloodVessel/imagesTs -o /home/pcallec/nnUNet_blood-vessel/results/test_pred/v3 -t 500 -m 2d 2>&1 | tee output_pred_v3.txt

(time CUDA_VISIBLE_DEVICES=1 nnUNet_predict -i /home/pcallec/nnUNet_blood-vessel/results/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BloodVessel/imagesTs -o /home/pcallec/nnUNet_blood-vessel/results/test_pred/v3_rep -t 500 -m 2d ) 2>&1 | tee output_pred_v3_rep.txt

```
# Extra test

```bash
python convert_label_1_channel_extra_test.py -j convert_label_1_channel_extra_test.json
python structure_data_extra_test.py -j structure_data_extra_test.json

python Task501_BloodVessel_extra_test.py

nnUNet_plan_and_preprocess -t 501 --verify_dataset_integrity -pl3d None

CUDA_VISIBLE_DEVICES=1 
nnUNet_predict -i /home/pcallec/nnUNet_blood-vessel/results/nnUNet_raw_data_base/nnUNet_raw_data/Task501_BloodVessel/imagesTr/ -o /home/pcallec/nnUNet_blood-vessel/results/extra_test_pred/ -t 501 -m 2d 2>&1 | tee output_pred_extra_test.txt
```