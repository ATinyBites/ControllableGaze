# What we need is explicit controllability: Training 3D gaze estimator using only facial images

<p align="center">
  <img src="assets/teaser.gif" alt="example input output gif" width="600" />
</p>

<!-- This repository provides the codebase for training and evaluating a **3D gaze estimation model** using **facial images only**, with a focus on **explicit controllability**. -->

---

## 🧰 Environment Setup
**​​CUDA 11.8**​​ must be installed first.Then create conda environment and install required packages as follows:
```bash
# Clone repository
git clone https://github.com/ATinyBites/ControllableGaze.git --recursive
cd controllable-gaze

# Create conda environment
conda env create -f environment.yml
conda activate control-gaze

# ​​Install remaining dependencies
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 🏋️ Model Training

### 1. Data Preparation
We provide preprocessed versions of the ​​MPIIGaze​​ and ​​ColumbiaGaze​​ datasets for avatar training.​ 
- Download: [Google Drive](https://drive.google.com/drive/folders/1EFQOYQo4TY_ayZ86vNv3vgVA5kS6Pks0?usp=sharing) | [Baidu Drive](https://pan.baidu.com/s/1e8-i9tsxIuepmfS1f6QgaA?pwd=mrjj).

Then extract the `mpii.zip` and  `columbia.zip` into the `dataset` directory.

---
Our method depend on FLAME model. Please download [FLAME](https://flame.is.tue.mpg.de/download.php) assets to the following paths:

- FLAME 2023 (versions w/ jaw rotation) -> flame_model/assets/flame/flame2023.pkl
- FLAME Vertex Masks -> flame_model/assets/flame/FLAME_masks.pkl

### 2. Start Training
```bash
conda activate control-gaze
# Train avatars for mpiiFaceGaze
bash ./scripts/train_mpii.sh
# Or train avatars for columbiaGaze
bash ./scripts/train_columbia.sh
```
After training, you can find the trained avatar models in the `output` directory.
## 🎯 Model Inference
After training avatar models, you can generate gaze datasets and subsequently train a gaze estimator.
### 1. Generate Gaze Dataset
```bash
# Generate gaze dataset for mpiiFaceGaze
bash ./scripts/simulate_mpii.sh
# Or generate gaze dataset for columbiaGaze
bash ./scripts/simulate_columbia.sh
```
The generated gaze datasets are stored in the `synthetic_dataset` directory.
### 2. Train Gaze Estimator
```bash
# Train gaze estimator for mpiiFaceGaze
python train_estimator.py --data_dir=synthetic_dataset/mpii --ckpt_dir=ckpt/mpii
# Or train gaze estimator for columbiaGaze
python train_estimator.py --data_dir=synthetic_dataset/columbia --ckpt_dir=ckpt/columbia
```

## ✅ Gaze Evaluation
To evaluate gaze estimation performance, you can use the provided `test.py` script with our preprocessed test sets from MPIIGaze and Columbia datasets.([Google Drive](https://drive.google.com/drive/folders/1SY4uEDohWrlv2z6yXclIED8uuP6gjYgc?usp=sharing) | [Baidu Drive](https://pan.baidu.com/s/1hZRZm3OuNH_ltzvhaoD7Nw?pwd=bdn4)) .

## Acknowledgements
Our code is developed based on:
- https://github.com/ShenhanQian/GaussianAvatars
- https://github.com/xucong-zhang/ETH-XGaze

## Citation 
```
@InProceedings{Li_2025_ICCV,
    author    = {Li, Tingwei and Bao, Jun and Kuang, Zhenzhong and Liu, Buyu},
    title     = {What we need is explicit controllability: Training 3D gaze estimator using only facial images},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {11414-11424}
}
```