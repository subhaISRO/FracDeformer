# This repo is the implementation for the inference of RainDrop removal challenge CVPR NTIRE 2025


## Contents

The contents of this repository are as follows:

1. Requirements (given in requirement.txt)
2. Use `python setup.py develop --no_cuda_ext` to setup the modified basicsr for this particular project.
3. [Test](#Test)



---

## Dependencies

- Python
- Pytorch (1.13)
- scikit-image
- opencv-python
- Tensorboard
- einops

---



---

## Train

bash train.sh

---

## Test

Realblur pre-trained model is available at https://drive.google.com/drive/folders/1l_R8_2UKfiQP_BYrgcQrmCBSe_ogwL41?usp=drive_link

bash test.sh

Output images will be saved in ``` results/model_name/dataset_name/``` folder.

We measured PSNR using [official RealBlur test code](https://github.com/rimchang/RealBlur#evaluation). You can get the PSNR we achieved by cloning and following the RealBlur repository.

---

## Acknowledgment: 
This code is based on the [Restormer](https://github.com/swz30/Restormer) and [NAFNet](https://github.com/megvii-research/NAFNet)

