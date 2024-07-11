
# Traininig

## Dataset setup
Generally, two parts of data are needed for training, the original dataset, i.e., ScanNet and MegaDepth, and the offline generated dataset indices. The dataset indices store scenes, image pairs, and other metadata within each dataset used for training/validation/testing. For the MegaDepth dataset, the relative poses between images used for training are directly cached in the indexing files. However, the relative poses of ScanNet image pairs are not stored due to the enormous resulting file size.

### Download datasets
#### MegaDepth
We use depth maps provided in the [original MegaDepth dataset](https://www.cs.cornell.edu/projects/megadepth/) as well as undistorted images, corresponding camera intrinsics and extrinsics preprocessed by [D2-Net](https://github.com/mihaidusmanu/d2-net#downloading-and-preprocessing-the-megadepth-dataset). You can download them separately from the following links. 
- [MegaDepth undistorted images and processed depths](https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz)
    - Note that we only use depth maps.
    - Path of the download data will be referreed to as `/path/to/megadepth`
- [D2-Net preprocessed images](https://drive.google.com/drive/folders/1hxpOsqOZefdrba_BqnW490XpNX_LgXPB)
    - Images are undistorted manually in D2-Net since the undistorted images from MegaDepth do not come with corresponding intrinsics.
    - Path of the download data will be referreed to as `/path/to/megadepth_d2net`

### Download the dataset indices

You can download the required dataset indices from the [following link](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf).
After downloading, unzip the required files.
```shell
unzip downloaded-file.zip

# extract dataset indices
tar xf train-data/megadepth_indices.tar
tar xf train-data/scannet_indices.tar

# extract testing data (optional)
tar xf testdata/megadepth_test_1500.tar
tar xf testdata/scannet_test_1500.tar
```

### Build the dataset symlinks

We symlink the datasets to the `data` directory under the main project directory.

# megadepth
```shell
# -- # train and test dataset (train and test share the same dataset)
ln -sv /path/to/megadepth/phoenix /path/to/megadepth_d2net/Undistorted_SfM /path/to/FineFormer/data/megadepth/train
ln -sv /path/to/megadepth/phoenix /path/to/megadepth_d2net/Undistorted_SfM /path/to/FineFormer/data/megadepth/test
# -- # dataset indices
ln -s /path/to/megadepth_indices/* /path/to/FineFormer/data/megadepth/index
```

