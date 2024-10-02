## Install requirement

```
pip install requirement.txt
```

## Install dataset

```
python main.py --make_dataset
```

## Model avaiable

- [x] SingleFrame: single_frame
- [x] CNNLSTM: cnn_lstm
- [x] LateFusion: late_fusion
- [x] EarlyFusion: early_fusion
- [x] S3D: s3d
- [x] VideoVisonTransformer: vivit

## Training

```
!python main.py --train --model vivit --epochs 5 --batch_size 4 --num_frames 900 --target_size 112 --num_classes 4
```
