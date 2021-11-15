## Data Preparation
Generate IDN labels either with [`Learning with Feature-Dependent Label Noise: A Progressive Approach (ICLR 2021)`](https://github.com/pxiangwu/PLC) or 
[`Beyond Class-Conditional Assumption: A Primary Attempt to Combat Instance-Dependent Label Noise (AAAI 2021)`](https://github.com/chenpf1025/IDN). Then move the label file to `labels` folder. Make sure that it follows the structure of first link when using second link to create noise. 

## Training
Simply run the command:
```
python train.py --noise_label_file labels.npy
```
With parameters of:
- `noise_label_file`: the `.npy` file for IDN noisy labels.
- `nepoch`: the number of training epochs.
- `noise_type`: the CCN noise type (`uniform` or `asymmetric`).
- `noise_rate`: the noise rate for CCN noise.
- `seed`: the random seed.
