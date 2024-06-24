# Beyond Global Cues: Unveiling the Power of Fine Details in Image Matching

## Introduction
The official code of our ICME 2024 paper.

## Dependencies
* Python 3 = 3.8.12
* PyTorch = 1.9.1

### Run the demo of our method
```sh
./demo.py
```
The default will resize images to `640x640`.

### Evaluation on MegaDepth
```shell
# with shell script
bash ./scripts/reproduce_test/outdoor_ds.sh
```

### Evaluation on YFCC100M
The script for evaluation on YFCC100M can be found in [superglue](https://github.com/magicleap/SuperGluePretrainedNetwork).

Actually, in our paper, some parameters has not been carefully tuned. When the RANSAC threshold is set to 0.3, the performance of our proposed method can be boosted again:
```txt
Evaluation Results (mean over 15 pairs):
AUC@5    AUC@10  AUC@20  Prec
42.55    61.69   76.75   93.87 
```

### A note on correction

In Table 1 and Table 2, the AUC scores of SP+OANet and SP+SGMNet are falsely cited. The same story goes for our [another work](https://ieeexplore.ieee.org/document/10485434). Here we make the correction:

In Table 1, the AUC scores of SuperPoint+SGMNet are corrected to 40.5 , 59.0 , 73.6

In Table 2, the AUC scores of OANet are corrected to 26.82 , 45.04 , 62.17

Notice that this is just a copy/paste mistake and has no influence about the ranking of tables and the conclusion of papers. 

## Citation
If you use any ideas from the paper or code from this repo, please consider cite our paper through IEEE Xplore.
