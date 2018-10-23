# TGS Salt Identification Challenge

## Data download

    kaggle c download --competition=tgs-salt-identification-challenge --path=input
    unzip input/test.zip -d input/test
    unzip input/train.zip -d input/train

## TODO



## memo

- cross-pseudo-labeling (+0.010)
- R,G=input B=depth (-0.010)
- gate (-0.006)
- mixup (-0.002)
- lovasz loss elu+1 with mixup (-0.002)
- multiple output (bin, mask) (-0.005)
- binary upsampling hypercolumn (+0.001)
- tail residual block (+0.001)
- scse block (+0.004)
- hypercolumn (+0.006)
- 112 == 224 (+0.001)
- padding > resize (+0.004)
- lovasz > `lovasz*0.9 + BCE*0.1` (+0.003)
- lovasz > BCE (+0.010)
- stack.py: bin-ir2 (+0.001)

