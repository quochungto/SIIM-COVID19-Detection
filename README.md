# 11th place Solution for SIIM COVID 19 Detection Challenge - Detection Part

## Solution Overview
**Final result**

| Category | Public LB (1/6 mAP) | Private LB (1/6 mAP)|
| --- | --- | --- |
| none | 0.134 | -- |
| opacity | 0.100 | -- |

**Cross-validation**

Stratified K Fold by StudyID

**None class prediction**

`none_probbility = np.prod(1 - box_conf_i)`

**Modeling**

**Detectors trained with competition train data only**

| backbone | image size | batch size | epochs | TTA | iou | conf | CV opacity (mAP) | CV none (mAP) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VFNetr50| 640 | 8 | 35 | Y | 0.5 | 0.001 | 0.48358 | 0.23121 |
| Yolov5m\* | 1024 | 8 | 35 | Y | 0.5 | 0.001 | 0.48148 | 0.76216 |
| Yolov5x\* | 640 | 8 | 35 | Y | 0.5 | 0.001 | 0.50930| -- |
| Yolov5x | 512 | 8 | 35 | Y | 0.5 | 0.001 | 0.51690 | 0.78192 |
| Yolov5l6 | 512 | 8 | 35 | Y | 0.5 | 0.001 | 0.51650 | 0.78190 |
| Yolov5x6 | 512 | 8 | 35 | Y | 0.5 | 0.001 | 0.51754 | 0.77820 |

\*: trained with different hyperparameter config

**Detectors trained with pseudo data**

| backbone | image size | batch size | epochs | TTA | iou | conf | CV opacity (mAP) | CV none (mAP) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Yolov5x | 512| 8 | 50 | Y | 0.5 | 0.001 | 0.53870 | 0.79028 |

**Pseudo labels - training process**

Datasets: Public test set + BIMCV + RICORD
- For BIMCV, the dataset contains a lot of images which are taken for the left/right side of human body. In order to reduce noise, my teammate @joven1997 and I manually removed them from the dataset. And since both training and test data in this competition are drawn from this dataset, to avoid leakage in validation, I removed all of the duplicate images and images that have the same StudyID with these duplicates.

Making pseudo labels:
- Label images with `none_probability > 0.6` as none class images
- For those have `none_probability <= 0.6`, keep boxes with confident > 0.095
These thresholds are chosen in order to maximize the f1 score.

Training:
All datasets are merged together and used to train with the same procedure as without pseudo data.

**Post-processing**	

- Weighted boxes fusion with `iou_thr=0.6` and `conf_thr=0.0001` as boxes fusion method
- `box_conf = box_conf**0.84 * (1 - none_probability)**0.16`
- `none_probability = none_probability*0.5 + negative_probability*0.5`
- `negative_probability = none_probability*0.3 + negative_probability*0.7`

**Final Submission**

For final submission, we used Yolotrs-384 + Yolov5x-640 + Yolov5x-512-pseudo labels, all with TTA.
