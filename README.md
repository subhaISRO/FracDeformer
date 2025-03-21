# This repo is the implementation for the inference of RainDrop removal challenge CVPR NTIRE 2025


## Contents

The contents of this repository are as follows:

1. [Requirements](#Dependencies)
2. [Test](#Test)
3. [Acknowledgements](#Acknowledgements)

---

## Dependencies

- Python
- Pytorch (1.13)
- scikit-image
- opencv-python
- Tensorboard
- einops

---

## Test

The checkpoints are available at: https://drive.google.com/file/d/10n_SiRSxjgog4nQY2ldktmQKCpM_Ezsg/view?usp=sharing

Download the checkpoints, uncompressed it and keep them in the `checkpoins` directory

Before running the inference, use `python setup.py develop --no_cuda_ext` to setup the modified basicsr for this particular project.

To run the inference code run the following line of python code

```python test_v2.py --input_dir [path to the inference images] --result_dir [where to save the predicted images, default: results] --blur_weights checkpoints/net_g_blur_removal.pth --drop_weights checkpoints/net_g_drop_removal.pth --dataset RainDrop_removal```

Output images will be saved in ``` results/model_name/dataset_name/``` folder.

---

## Acknowledgment: 
This code is based on the [FFTformer](https://github.com/kkkls/FFTformer) and [NAFNet](https://github.com/megvii-research/NAFNet)

