# TGS Salt Identification Challenge

## Data download

    kaggle c download --competition=tgs-salt-identification-challenge --path=input
    unzip input/test.zip -d input/test
    unzip input/train.zip -d input/train

## TODO

- cross-pseudo-labeling: XyをNoneにしたデータを一定数入れといてgenで処理。cv-index +1のモデルでやると、2,3週はいけるはず。


## memo

- 112 == 224
- padding > resize (+0.004)
- lovasz > BCE (+0.010)

