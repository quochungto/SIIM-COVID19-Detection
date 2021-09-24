import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score
from mean_average_precision import MetricBuilder

def str2boxes(s, with_prob=True):
    bboxes = s.strip().split()

    if with_prob:
        # [x_min y_min x_max y_max class_id prob]
        bboxes = [bboxes[6*idx+2:6*idx+6] + [0] + [bboxes[6*idx+1]] for idx in range(len(bboxes)//6)]
    else:
        # [x_min y_min x_max y_max class_id difficult crownd]
        bboxes = [bboxes[6*idx+2:6*idx+6] + [0, 0, 0] for idx in range(len(bboxes)//6)]

    bboxes = np.array(bboxes, dtype=np.float32)#.tolist()
    return bboxes


def train_csv2voc(df):
    all_bboxes = []

    for i, row in df.iterrows():
        if row['image_label'] == 'none 1 0 0 1 1':
            all_bboxes.append(np.array([], dtype=np.float32))
        else:
            bboxes = str2boxes(row['image_label'], with_prob=False)
            all_bboxes.append(bboxes)

    return all_bboxes


def sub_csv2voc(df):
    all_bboxes = []

    for i, row in df.iterrows():
        bboxes = str2boxes(row['PredictionString'], with_prob=True)
        all_bboxes.append(bboxes)

    return all_bboxes


def voc_map(pred, truth, iou_thresholds=0.5):
    map_fn = MetricBuilder.build_evaluation_metric('map_2d', async_mode=True, num_classes=1)

    for p, t in zip(pred, truth):
        map_fn.add(p, t)

    result = map_fn.value(iou_thresholds=iou_thresholds)['mAP']

    return result

def map_2cls(df_truth, df_pred, iou_thresholds=0.5, post_process=False):

    #----make sure orders of ids in both dataframe
    if not df_truth.loc[0, 'id'].endswith('image'):
        df_truth['id'] = df_truth['id'] + '_image'
    if not df_pred.loc[0, 'id'].endswith('image'):
        df_pred['id'] = df_pred['id'] + '_image'

    if len(df_pred) > len(df_truth):
        df_pred = df_pred.set_index(['id']).loc[df_truth['id']].reset_index()
    else:
        df_truth = df_truth.set_index(['id']).loc[df_pred['id']].reset_index()
    #----

    #----opacity
    opacity_truth = train_csv2voc(df_truth)
    opacity_pred = sub_csv2voc(df_pred)
    # get all box's probs
    opacity_prob = [op_pred[:,-1] if len(op_pred) > 0 else np.array([0.]) for op_pred in opacity_pred]

    if not post_process:
        opacity_map = voc_map(opacity_pred, opacity_truth, iou_thresholds=iou_thresholds)
    #----

    #----none
    none_truth = (df_truth['image_label'] == 'none 1 0 0 1 1').values

    none_pred = extract_none_probs(opacity_prob)
    none_pred = np.array(none_pred).reshape(-1, 1)

    none_map = average_precision_score(none_truth, none_pred)
    #----

    #----post-processing
    if post_process:
        for i, (o_p, n_p) in enumerate(zip(opacity_pred, none_pred.reshape(-1))):
            opacity_pred[i][:,-1] = o_p[:,-1]*((1 - n_p)**0.2)

        opacity_map = voc_map(opacity_pred, opacity_truth, iou_thresholds=iou_thresholds)

    return opacity_map, none_map


def validate_detection_one_fold(pred_path, truth_path):
    df_truth = pd.read_csv(truth_path)
    df_pred = pd.read_csv(pred_path)
    
    #----
    if not df_truth.loc[0, 'id'].endswith('image'):
        df_truth['id'] = df_truth['id'] + '_image'
    if not df_pred.loc[0, 'id'].endswith('image'):
        df_pred['id'] = df_pred['id'] + '_image'
    #----

    o_map, n_map = map_2cls(df_truth, df_pred, post_process=False)

    return o_map, n_map
