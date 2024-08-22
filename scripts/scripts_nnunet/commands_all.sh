nnUNet_train 2d nnUNetTrainerV2 500 0 2>&1 | tee output_new_fold0.txt 
nnUNet_train 2d nnUNetTrainerV2 500 1 2>&1 | tee output_new_fold1.txt 
CUDA_VISIBLE_DEVICES=1 nnUNet_train 2d nnUNetTrainerV2 500 2 2>&1 | tee output_new_fold2.txt 
CUDA_VISIBLE_DEVICES=1 nnUNet_train 2d nnUNetTrainerV2 500 3 2>&1 | tee output_new_fold3.txt 
CUDA_VISIBLE_DEVICES=0 nnUNet_train 2d nnUNetTrainerV2 500 4 2>&1 | tee output_new_fold4.txt 
