# Beyond Global Cues: Unveiling the Power of Fine Details in Image Matching

## Introduction
The official code of our ICME 2024 paper.

## Dependencies
* Python 3 = 3.8.12
* PyTorch = 1.9.1

### Run the demo of our method
```sh
python  ./demo.py
```
The default will resize images to `640x640`.

### Weights

Weights can be found [here](https://pan.baidu.com/s/1W-Ame3A3s378JKSB2bOjNw?pwd=38n7)

### Data preparation

For details of data preparation, please refer to [loftr](https://github.com/zju3dv/LoFTR).

### Evaluation on MegaDepth
```shell
# with shell script
bash ./scripts/reproduce_test/outdoor_ds.sh
```

### Evaluation on YFCC100M
For the script for evaluation on YFCC100M, you can refer to [superglue](https://github.com/magicleap/SuperGluePretrainedNetwork).

When the RANSAC threshold is set to 1.0, you can reproduce results in paper.

When the RANSAC threshold is set to 0.3, the performance of our proposed method can be boosted again:
```txt
Evaluation Results (mean over 15 pairs):
AUC@5    AUC@10  AUC@20  Prec
42.55    61.69   76.75   93.87 
```

### About training

If you want to train yourself, please
```sh
python  ./train.py
```

Due to the limited computational resources, the images are resized to `640x640` during training. You are suggested to use the larger sizes since many works have proven that the large sizes can improve the performance.

### A note on correction

In Table 1 and Table 2, the AUC scores of SP+OANet and SP+SGMNet are not accurate. The same story goes for our [another work](https://ieeexplore.ieee.org/document/10485434). Here we make the correction:

In Table 1, the AUC scores of SuperPoint+SGMNet are corrected to 40.5 , 59.0 , 73.6

In Table 2, the AUC scores of OANet are corrected to 26.82 , 45.04 , 62.17

Notice that this is just a copy/paste mistake and has no influence about evaluation and conclusions of our papers. 

## Citation
If you use any ideas from the paper or code from this repo, please consider cite our paper through IEEE Xplore.
