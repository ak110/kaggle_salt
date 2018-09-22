# TGS Salt Identification Challenge

## Data download

    kaggle c download --competition=tgs-salt-identification-challenge --path=input
    unzip input/test.zip -d input/test
    unzip input/train.zip -d input/train

## TODO

- cross-pseudo-labeling: XyをNoneにしたデータを一定数入れといてgenで処理。cv-index +1のモデルでやると、2,3週はいけるはず。


## memo

- binary upsampling hypercolumn (+0.001)
- tail residual block (+0.001)
- scse block (+0.004)
- hypercolumn (+0.006)
- 112 == 224 (+0.001)
- padding > resize (+0.004)
- lovasz > `lovasz*0.9 + BCE*0.1` (+0.003)
- lovasz > BCE (+0.010)
- stack.py: bin-ir2 (+0.001)

