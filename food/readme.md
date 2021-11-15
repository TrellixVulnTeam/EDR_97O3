## Data Preparation
Download [Food-101N](https://kuanghuei.github.io/Food-101N/) and [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) and put them in `data` folder. Then run the following command:
```
python food_dataset.py
```

## Training
Simply run the command:
```
python train.py
```
With parameters of:
- `nepoch`: the number of training epochs.
- `seed`: the random seed.
