# 11th place Solution for SIIM-FISABIO-RSNA COVID-19 Detection Challenge - Detection Part
This documentation outlines how to reproduce Detection part result of the 11th place solution by the "∫ℳΓϒℏ" team for the [COVID19 Detection Competition](https://www.kaggle.com/c/siim-covid19-detection/overview) on Kaggle hosted by SIIM-FISABIO-RSNA. 

[Code for reproducing Classification part result](https://github.com/ChenYingpeng/pl-siim-covid19-detection)

## 1. Solution Overview
[Solution overview for Classification part](https://www.kaggle.com/c/siim-covid19-detection/discussion/263701)

Below is the overview for Detection part solution

### 1.1. Final result

| Category | Public LB (1/6 mAP@.5) | Private LB (1/6 mAP@.5)|
| --- | --- | --- |
| none | 0.134 | -- |
| opacity | 0.100 | -- |

### 1.2. Cross-validation

Stratified K Fold by StudyID

### 1.3. None-class prediction

`none_probbility = np.prod(1 - box_conf_i)`

### 1.4. Training models

**Detectors trained with competition train data only**

| backbone | image size | batch size | epochs | TTA | iou | conf | CV opacity (mAP@.5) | CV none (mAP@.5) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VFNetr50| 640 | 8 | 35 | Y | 0.5 | 0.001 | 0.48358 | 0.23121 |
| Yolov5m\* | 1024 | 8 | 35 | Y | 0.5 | 0.001 | 0.48148 | 0.76216 |
| Yolov5x\* | 640 | 8 | 35 | Y | 0.5 | 0.001 | 0.50930| -- |
| Yolov5x | 512 | 8 | 35 | Y | 0.5 | 0.001 | 0.51690 | 0.78192 |
| Yolov5l6 | 512 | 8 | 35 | Y | 0.5 | 0.001 | 0.51650 | 0.78190 |
| Yolov5x6 | 512 | 8 | 35 | Y | 0.5 | 0.001 | 0.51754 | 0.77820 |
| YoloTrs | 512 | 32 | 40 | Y | 0.5 | 0.001 | 0.51343 | 0.776458 |

<sub>\*: trained with different hyperparameter config<sub>

**Detectors trained with pseudo data**

| backbone | image size | batch size | epochs | TTA | iou | conf | CV opacity (mAP@.5) | CV none (mAP@.5) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Yolov5x | 512| 8 | 50 | Y | 0.5 | 0.001 | 0.53870 | 0.79028 |

### 1.5. Pseudo labels

**Datasets**: Public test set + BIMCV + RICORD
- For BIMCV, the dataset contains a lot of images which are taken for the left/right side of human body. In order to reduce noise, we manually removed them from the dataset. And since both training and test data in this competition are drawn from this dataset, to avoid leakage in validation, we removed all of the duplicate images and images that have the same StudyID with these duplicates.

**Making pseudo labels**
- Label images with `none_probability > 0.6` as none class images
- For those have `none_probability <= 0.6`, keep boxes with confident > 0.095
These thresholds are chosen in order to maximize the f1 score.

**Training**
All datasets are merged together and used to train with the same procedure as without pseudo data.

### 1.6. Post-processing

- Weighted boxes fusion with `iou_thr=0.6` and `conf_thr=0.0001` as boxes fusion method
- `box_conf = box_conf**0.84 * (1 - none_probability)**0.16`
- `none_probability = none_probability*0.5 + negative_probability*0.5`
- `negative_probability = none_probability*0.3 + negative_probability*0.7`

### 1.7. Final submission

For final submission, we used Yolotrs-384 + Yolov5x-640 + Yolov5x-512-pseudo labels, all with TTA.

## 2. Installation
- Ubuntu 20.04.01 LTS
- Python 3.8
- python packages are detailed separately in [requirements](https://github.com/dungnb1333/SIIM-COVID19-Detection/blob/main/requirements.txt)
```
$ vitualenv --python=python3.8 envs
$ source envs/bin/activate
$ pip install -r requirements.txt
```

## 3. Dataset
All required datasets will be automatically downloaded via command:
```
$ ./download_datasets.sh
```
The downloaded datasets will be placed in directory ```./dataset```, including:

- ```fold-split-siim``` fold split data for the train dataset using Stratified K Fold
- ```1024x1024-png-siim``` competition train dataset
- ```metadatasets-siim``` metadata for the train dataset
- ```image-level-psuedo-label-metadata-siim``` metadata for the pseudo label datasets
- ```ricord-covid19-xray-positive-tests``` RICORD COVID-19 X-ray positive tests dataset
- ```covid19-posi-dump-siim``` BIMCV COVID-19 dataset

## 4. Train models
Navigate your working directory into ```./src```
```
$ cd ./src
```
### 4.1. Yolo variants
### 4.1.1. Train
Train detectors including ```yolov5s, yolov5m, yolov5l, yolov5x, yolov5s6, yolov5m6, yolov5l6, yolov5x6, yolotrs```
```
# Train a Yolo-Transfomer-s for 3 epochs on folds 0 and 1
$ python ./detection/yolo/train.py --weight yolotrs --folds 0,1 --img 640 --batch 16
```
To train on both train data and pseudo-labeled data, add flag ```--pseduo path/to/csv``` to the end of the above command.
All checkpoints will be saved at ```./result/yolo/checkpoints```
### 4.1.2. Predict
```
$ python ./detection/yolo/infer.py \
$ -ck ../result/yolo/checkpoints/best0.pt \ # paths to model checkpoints
$     ../result/yolo/checkpoints/best1.pt \
$ --iou 0.5 \                               # box fusion iou threshold 
$ --conf 0.0001 \                           # box fusion skip box threshold
$ --mode remote \                           # 'local' mode for evaluating on validation dataset,
                                              'remote' mode for predicting on test dataset
                                              'pseudo' mode for predicting on external datasets
$ --image 614 \
$ --batch 32
```
Ouput .csv files will be saved at ```./result/yolo/submit```

### 4.2 VFNet
### 4.2.1. Train
Train detectors including ```vfnetr50, vfnetr101```
```
# Train a VFNetr50 for 3 epochs on folds 0 and 1
$ python ./detection/mmdet/train.py --weight vfnetr50 --folds 0,1 --img 640 --batch 16
```
All checkpoints will be saved at ```./result/mmdet/checkpoints```
### 4.2.2. Predict
```
$ python ./detection/yolo/infer.py \
$ -ck ../result/mmdet/checkpoints/best0.pt \ # paths to model checkpoints
$     ../result/mmdet/checkpoints/best1.pt \
$ --iou 0.5 \                               # box fusion iou threshold 
$ --conf 0.0001 \                           # box fusion skip box threshold
$ --mode remote \                           # 'local' mode for evaluating on validation dataset,
                                              'remote' mode for predicting on test dataset
$ --image 614 \
$ --batch 32
```
Ouput .csv files will be saved at ```./result/mmdet/submit```

## 5. Generate pseudo labels
### 5.1. Predict
```
$ python ./detection/yolo/infer.py \
$ -ck ../result/mmdet/checkpoints/best0.pt \ # paths to model checkpoints
$     ../result/mmdet/checkpoints/best1.pt \
$ --iou 0.5 \                                # box fusion iou threshold 
$ --conf 0.0001 \                            # box fusion skip box threshold
$ --mode pseudo \                            # 'local' mode for evaluating on validation dataset,
                                               'remote' mode for predicting on test dataset
                                               'pseudo' mode for predicting on external datasets
$ --image 614 \
$ --batch 32
```
Ouput .csv files will be saved at ```./result/pseudo/predict```

### 5.2. Hard-label data
```
$ python ./detection/make_pseudo.py \
$ -paths ../result/best0.csv \ # paths to predicted csv files
$        ../result/best1.csv \
$ -ws 2 1 \                    # ensemble weights in same order as -paths
$ --iou 0.6 \                  # box fusion iou threshold
$ --conf 0.001 \               # box fusion skip box threshold
$ --none 0.6 \                 # theshold for hard-labeling images as none-class
$ --opacity 0.095              # threshold for hard-labeling images as opacity-class
```
Ouput .csv files will be saved at ```./result/pseudo/labeled```

## 6. Ensemble & Post-process & Final submission
Final submission file will be named ```submission.csv``` and saved at ```.\result\submission```.
```
$ python ./post_processing/postprocess.py \
$ -study ../result/submit/study/best0.csv \ # paths to study-level csv files
$        ../result/submit/study/best1.csv \	
$ -image ../result/submit/image/best0.csv \ # paths to image-level csv files
$        ../result/submit/image/best1.csv \
$ -sw 1 2 \                                 # study-level ensemble weights in same order as -study
$ -iw 1 1 \                                 # image-level ensemble weights in same order as -image
$ --iou 0.6 \                               # box fusion iou threshold
$ --conf 0.001                              # box fusion skip box threshold
```

## 7. Softwares & Resources
[Pytorch](https://github.com/pytorch/pytorch)\
[Albumentations](https://github.com/albumentations-team/albumentations)\
[YoloV5](https://github.com/ultralytics/yolov5)\
[MMDetection](https://github.com/open-mmlab/mmdetection)\
[Weighted Boxes Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
