## Data Preparation
Download Animal-10N ([Link](https://dm.kaist.ac.kr/datasets/animal-10n/)).
Rename the folders to `train` and `test`, then put both image in another folder called `data` before running the code. Also transfer 10% of clean training data (manually check the labels) `val` folder.

## Training
Simply run the command:
```
python train.py
```

With parameters of:
- `nepoch`: the number of training epochs.
- `seed`: the random seed.
