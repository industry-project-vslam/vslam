### ScanNet++

1. Download the [dataset](https://kaldir.vc.in.tum.de/scannetpp/), extract RGB frames and masks from the iPhone data following the [official instruction](https://github.com/scannetpp/scannetpp). 

2. Preprocess the data with the following command:

```bash
python datasets_preprocess/preprocess_scannetpp.py \
--scannetpp_dir $SCANNETPP_DATA_ROOT\
--output_dir data/scannetpp_processed
```

the processed data will be saved at `./data/scannetpp_processed`

> We only use ScanNetpp-V1 (280 scenes in total) to train and validate our SLAM3R models now. ScanNetpp-V2 (906 scenes) is available for potential use, but you may need to modify the scripts for certain scenes in it.

### Aria Synthetic Environments

For more details, please refer to the [official website](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset)

1. Prepare the codebase and environment
```bash
mkdir data/projectaria 
cd data/projectaria
git clone https://github.com/facebookresearch/projectaria_tools.git -b 1.5.7
cd -
conda create -n aria python=3.10
conda activate aria
pip install projectaria-tools'[all]' opencv-python open3d
```

2. Get the download-urls file [here](https://www.projectaria.com/datasets/ase/) and place it under .`/data/projectaria/projectaria_tools`. Then download the ASE dataset:
```bash
cd ./data/projectaria/projectaria_tools
python projects/AriaSyntheticEnvironment/aria_synthetic_environments_downloader.py \
--set train \
--scene-ids 0-499 \
--unzip True \
--cdn-file aria_synthetic_environments_dataset_download_urls.json \
--output-dir $SLAM3R_DIR/data/projectaria/ase_raw 
```

> We only use the first 500 scenes to train and validate our SLAM3R models now. You can leverage more scenes depending on your resources.

4. Preprocess the data.
```bash
cp ./datasets_preprocess/preprocess_ase.py ./data/projectaria/projectaria_tools/
cd ./data/projectaria
python projectaria_tools/preprocess_ase.py 
```
The processed data will be saved at `./data/projectaria/ase_processed`


### CO3Dv2
1. Download the [dataset](https://github.com/facebookresearch/co3d)

2. Preprocess the data with the same script as in [DUSt3R](https://github.com/naver/dust3r?tab=readme-ov-file), and place the processed data at `./data/co3d_processed`. The data consists of 41 categories for training and 10 categories for validation.

