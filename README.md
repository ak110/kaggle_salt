# TGS Salt Identification Challenge

## Data download

    kaggle c download --competition=tgs-salt-identification-challenge --path=input
    unzip input/test.zip -d input/test
    unzip input/train.zip -d input/train

## Setup

- pip install pydensecrf

## TODO

- TTA (LR?)
- cross-pseudo-labeling: XyをNoneにしたデータを一定数入れといてgenで処理。cv-index +1のモデルでやると、2,3週はいけるはず。
- ensemble
    - predict.pyが最初にキャッシュを削除してからキャッシュディレクトリありで呼び出す感じがよい
