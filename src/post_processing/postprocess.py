import sys
sys.path.append('.')

# stdlib
import os
from glob import glob
from tqdm.auto import tqdm
import json
import pickle
from collections import defaultdict
import time
import argparse

# numlib
import numpy as np
import pandas as pd

#from include import *
from utils.file import Logger
from utils.him import downsize_boxes, upsize_boxes

def am_mean(data, ws):
    return np.sum([d*w for d, w in zip(data, ws)])/np.sum(ws)

def gm_mean(data, ws):
    return np.prod([d**w for d, w in zip(data, ws)])**(1./np.sum(ws))

def am_gm_mean(data, ws):
    return 0.5*(am_mean(data, ws) + gm_mean(data, ws))

def str2boxes_image(s, with_none=False):
    """
    ouput: [prob x_min y_min x_max y_max]
    range x,y: [0, +inf]
    """ 
    s = s.strip().split()
    s = np.array([s[6*idx+1:6*idx+6] for idx in range(len(s)//6) \
            if s[6*idx] == 'opacity' or with_none]).astype(np.float32)
    if len(s) == 0: print('Warning: image without box!')
    return s


def str2boxes_df(df, with_none=False):
    return [str2boxes_image(row['PredictionString'], with_none=with_none) \
            for _, row in df.iterrows()]


def boxes2str_image(boxes):
    if len(boxes) == 0:
        return ''
    return ' '.join(np.concatenate([[['opacity']]*len(boxes), boxes], \
            axis=1).reshape(-1).astype('str'))


def boxes2str_df(boxes, image_ids=None):
    strs = [boxes2str_image(bs) for bs in boxes]
    if image_ids is None:
        return strs
    return pd.DataFrame({'id': image_ids, 'PredictionString': strs})


def check_num_boxes_per_image(df=None, csv_path=None):
    assert df is not None or csv_path is not None
    if df is None:
        df = pd.read_csv(csv_path)
    df_image = df[df['id'].apply(lambda x: x.endswith('image'))].reset_index(drop=True)
    all_boxes = str2boxes_df(df_image, with_none=False)
    all_boxes = [boxes for boxes in all_boxes if len(boxes) > 0 ]
    return np.concatenate(all_boxes).shape[0] / len(df_image)


def extract_none_probs(opacity_probs):
    none_probs = []

    for image_probs in opacity_probs:
        none_prob = np.prod(1 - np.array(image_probs))
        none_probs.append(none_prob)

    return none_probs


def filter_rows(df, mode):
    assert mode in ['study', 'image']
    df = df.copy()
    df = df[df['id'].apply(lambda x: x.endswith(mode))].reset_index(drop=True)
    return df


def ensemble_image(dfs, df_meta, mode='wbf', \
        iou_thr=0.5, skip_box_thr=0.001, weights=None):
    
    df_meta = filter_rows(df_meta, mode='image')
    dfs = [filter_rows(df, mode='image') for df in dfs]
    
    image_ids, prediction_strings, all_scores = [], [], []
    num_boxes = 0
    
    for i, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
        image_id = row['id']
        
        s = []
        for df in dfs:
            if np.sum(df['id']==image_id) > 0:
                ss = df.loc[df['id']==image_id, 'PredictionString'].values[0]
                if type(ss) == str:
                    s.append(ss)
                else:
                    s.append('')
            else:
                s.append('')
        
        boxes, scores, labels = [], [], []
        for ss in s:
            boxes_, scores_, labels_ = [], [], []
                
            ss = str2boxes_image(ss, with_none=False)
            
            if len(ss) > 0:
                labels_ = [0]*len(ss)
                scores_ = ss[:, 0].tolist()
                boxes_ = downsize_boxes(ss[:, 1:], row['w'], row['h'])
                
            labels.append(labels_)
            boxes.append(boxes_)
            scores.append(scores_)
        
        if mode == 'wbf':
            boxes, scores, labels = weighted_boxes_fusion(boxes,
                                                          scores,
                                                          labels,
                                                          iou_thr=iou_thr,
                                                          weights=weights,
                                                          skip_box_thr=skip_box_thr)
        elif mode == 'nms':
            boxes_, scores_, labels_, weights_ = [], [], [], []
            
            for j, b in enumerate(boxes):
                if len(b) > 0:
                    boxes_.append(b)
                    scores_.append(scores[j])
                    labels_.append(labels[j])
                    if weights is not None:
                        weights_.append(weights[j])

            if weights is None:
                weights_ = None

            boxes, scores, labels = nms(boxes_,
                                        scores_,
                                        labels_,
                                        iou_thr=iou_thr,
                                        weights=weights_)


        if len(boxes) == 0:
            image_ids.append(image_id)
            prediction_strings.append('')
            print('Warning: no box found after boxes fusion!')
            continue

        num_boxes += len(boxes)
        all_scores.append(scores)

        boxes = upsize_boxes(boxes, row['w'], row['h'])
        
        s = []
        for box, score, label in zip(boxes, scores, labels):
            s.append(' '.join(['opacity', str(score), ' '.join(box.astype(str))]))

        image_ids.append(image_id)
        prediction_strings.append(' '.join(s))

    df_pred = pd.DataFrame({'id': image_ids, 'PredictionString': prediction_strings})

    return df_pred, num_boxes, np.concatenate(all_scores).tolist()

def ensemble_study(dfs, weights=None, mean='am'):
    dfs = [filter_rows(df, mode='study') for df in dfs]
    study_ids = dfs[0]['id'].values

    if weights is None:
        weights = [1.] * len(dfs)

    weights = np.array(weights) / np.sum(weights)

    ens_probs_am = np.zeros((len(study_ids), 4), dtype=np.float32)
    ens_probs_gm = np.ones((len(study_ids), 4), dtype=np.float32)

    for df, w in zip(dfs, weights):
        df = df[df['id'].apply(lambda x: x.endswith('study'))].reset_index(drop=False)

        for i, id_ in enumerate(study_ids):
            s = df.loc[df['id']==id_, 'PredictionString'].values[0]
            preds = s.strip().split()
            for idx in range(len(preds)//6):
                ens_probs_am[i, cls_map[preds[6*idx]]] += float(preds[6*idx + 1]) * w
                ens_probs_gm[i, cls_map[preds[6*idx]]] *= float(preds[6*idx + 1]) ** w

    # apply different ensemble methods
    if mean == 'am':
        ens_probs = ens_probs_am
    elif mean == 'gm':
        ens_probs = ens_probs_gm
    elif mean == 'am_gm':
        ens_probs = 0.5*(ens_probs_am + ens_probs_gm)

    df = pd.DataFrame({'id': study_ids})
    df[class_names] = ens_probs
    df['PredictionString'] = df.apply(lambda row: \
            f'negative {row["negative"]} 0 0 1 1 typical {row["typical"]} 0 0 1 1 \
            indeterminate {row["indeterminate"]} 0 0 1 1 atypical {row["atypical"]} 0 0 1 1', \
            axis=1)

    df = df[['id', 'PredictionString']]

    return df


def extract_negative_prob(df, std2img):
    """
    Args: 
        df: study-level df
        std2img: dict maps from study_id to image_id
    Returns: 
        df with image-level ids and mapped negative probabilities
    """
    df = filter_rows(df, mode='study')

    image_ids, negative_probs = [], []
    for study_id, img_ids in std2img.items():
        s = df.loc[df['id']==study_id + '_study', 'PredictionString'].values[0]
        s = s.strip().split()
        for idx in range(len(s)//6):
            if s[6*idx] == 'negative':
                neg_prob = float(s[6*idx + 1])
                break
        image_ids.extend([img_id + '_image' for img_id in img_ids])
        negative_probs.extend([neg_prob]*len(img_ids))

    return pd.DataFrame({'id': image_ids, 'negative': negative_probs})


def postprocess_image(df_image, df_study, std2img, df_none=None, \
        none_cls_w=0., none_dec_w=0.5, neg_w=0.5, \
        detect_w=0.84, clsf_w=0.84):
    df_image = filter_rows(df_image, mode='image')
    df_study = filter_rows(df_study, mode='study')

    if df_none is None:
        none_cls_w = 0. 

    none_cls_w, none_dec_w, neg_w = \
            none_cls_w/(none_cls_w + none_dec_w + neg_w), \
            none_dec_w/(none_cls_w + none_dec_w + neg_w), \
            neg_w/(none_cls_w + none_dec_w + neg_w)

    detect_w, clsf_w = \
            detect_w/(detect_w + clsf_w), \
            clsf_w/(detect_w + clsf_w)

    df_negative = extract_negative_prob(df_study, std2img)
    
    df_image = df_image.merge(df_negative, on='id', how='left')
    if none_cls_w > 0.:
        df_image = df_image.merge(df_none, on='id', how='left')

    new_nones = []

    for i, row in df_image.iterrows():
        if row['PredictionString'] == 'none 1 0 0 1 1' \
                or row['PredictionString'] == '' \
                or type(row['PredictionString']) != str:

            df_image.loc[i, 'PredictionString'] = f'none {row["none"]} 0 0 1 1'
            #df_image.loc[i, 'new_none'] = row["none"]
            new_nones.append(row["none"])
            print('no opacity founded!')
            continue
        else:
            # extract none probabilities
            none_dec_prob = 1.
            bboxes = row['PredictionString'].strip().split()
            for idx in range(len(bboxes)//6):
                if bboxes[6*idx] == 'opacity':
                    none_dec_prob *= 1 - float(bboxes[6*idx + 1])

            # modify opacity boxes
            if none_cls_w > 0.:
                post_none_prob = none_cls_w*row["none"] + none_dec_w*none_dec_prob + neg_w*row["negative"]
            else:
                post_none_prob = none_dec_w*none_dec_prob + neg_w*row["negative"]

            for idx in range(len(bboxes)//6):
                if bboxes[6*idx] == 'opacity':
                    bboxes[6*idx + 1] = str(float(bboxes[6*idx + 1])**detect_w * (1 - post_none_prob)**clsf_w)

            df_image.loc[i, 'PredictionString'] = ' '.join(bboxes)

            # add none boxes
            df_image.loc[i, 'PredictionString'] += f' none {post_none_prob} 0 0 1 1'

            # act none probability for ensemble with negative in study-level
            if none_cls_w > 0.:
                new_nones.append(none_cls_w/(none_cls_w + none_dec_w))*row["none"] + \
                        (none_dec_w/(none_cls_w + none_dec_w))*none_dec_prob
            else:
                new_nones.append(none_dec_prob)

    df_none = pd.DataFrame({'id': df_image['id'].values, 'none': new_nones})

    return df_image, df_none

def postprocess_study(df, df_none, std2img, neg_w=0.7, none_w=0.3):
    """
    Args:
        df: study-level prediction
        df_none: image-level none probability
        std2img: dict maps from study_id to image_id
    """
    df = filter_rows(df, mode='study')
    df_none = filter_rows(df_none, mode='image')

    neg_w, none_w = \
        neg_w/(neg_w + none_w), \
        none_w/(neg_w + none_w)

    # extract none probability for each study
    study_ids, none_probs = [], []
    for study_id, image_ids in std2img.items():
        image_ids_ = [img_id + '_image' for img_id in image_ids]
        study_none_prob = df_none.loc[df_none['id'].isin(image_ids_), 'none'].mean()
        study_ids.append(study_id + '_study')
        none_probs.append(study_none_prob)

    df_study_none = pd.DataFrame({'id': study_ids, 'none': none_probs})
    df = pd.merge(df, df_study_none, on='id', how='left')

    for i, row in df.iterrows():
        #----modifiy negative probalibity
        bboxes = row['PredictionString'].strip().split()
        for idx in range(len(bboxes)//6):
            if bboxes[6*idx] == 'negative':
                bboxes[6*idx + 1] = str(neg_w*float(bboxes[6*idx + 1]) + none_w*float(row['none']))
                break

        df.loc[i, 'PredictionString'] = ' '.join(bboxes)

    df = df[['id', 'PredictionString']]
    return df

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('-study-csv', type=str, nargs='+', help='paths to study-level csv', required=True)# img-level paths
    parser.add_argument('-image-csv', type=str, nargs='+', help='paths to image-level csvrequired', required=True)# study-level paths
    parser.add_argument('-sw', type=float, nargs='+', help='study-level ensemble weights', required=True)# img-level weights
    parser.add_argument('-iw', type=float, nargs='+', help='image-level ensemble weights', required=True)# img-level weights
    parser.add_argument('-std2img', type=str, help='path to study2image pickle', required=True)# std2img dict
    parser.add_argument('-img2shape', type=str, help='path to image2shape pickle', required=True)# meta data path
    parser.add_argument('--iou-thr', type=float, default=0.6, help='boxes fusion iou threshold')# iou thres
    parser.add_argument('--conf-thr', type=float, default=0.0001, help='boxes fusion skip box threshold')# conf thes
    parser.add_argument('--none-csv', type=str, help='path to none csv in case of using seperate none probability')

    return parser.parse_args()

def main():
    t0 = time.time()
    
    opt = parse_opt()

    assert len(opt.study_csv) == len(opt.sw), f'len(study_csv) == {len(opt.study_csv)} != len(sw) == {opt.sw}'
    assert len(opt.image_csv) == len(opt.iw), f'len(image_csv) == {len(opt.image_csv)} != len(iw) == {opt.iw}'
    
    # logging
    log = Logger()
    os.makedirs('../logging', exist_ok=True)
    log.open(os.path.join('../logging', 'post_processing.txt'), mode='a')
    log.write('STUDY-LEVEL\n')
    log.write('weight\tpath\n')
    for p, w in zip(opt.study_csv, opt.sw):
        log.write('%.2f\t%s\n'%(w, p))
    log.write('\n')
    log.write('IMAGE-LEVEL\n')
    log.write('weight\tpath\n')
    for p, w in zip(opt.image_csv, opt.iw):
        log.write('%.2f\t%s\n'%(w, p))
    log.write('\n')
    log.write('iou_thr=%.4f,skip_box_thr=%.4f\n'%(opt.iou_thr, opt.conf_thr))

    # prepare data    
    dfs_study = [pd.read_csv(df_path) for df_path in opt.study_csv]
    dfs_image = [pd.read_csv(df_path) for df_path in opt.image_csv]
    with open(opt.std2img, 'rb') as f:
        std2img = pickle.load(f)

    with open(opt.img2shape, 'rb') as f:
        img2shape = pickle.load(f)
    ids, hs, ws = [], [], []
    for k, v in img2shape.items():
        ids.append(k + '_image')
        hs.append(v[0])
        ws.append(v[1])
    df_meta = pd.DataFrame({'id': ids, 'w': ws, 'h': hs})

    # post-process
    df_study = ensemble_study(dfs_study, weights=opt.sw)
    df_image = ensemble_image(dfs_image, df_meta, mode='wbf', \
            iou_thr=opt.iou_thr, skip_box_thr=opt.conf_thr, weights=opt.iw)[0]
    df_image, df_none = postprocess_image(df_image, df_study, std2img) 
    df_study = postprocess_study(df_study, df_none, std2img)

    df_sub = pd.concat([df_study, df_image], axis=0, ignore_index=True)
    df_sub = df_sub[['id', 'PredictionString']]
    df_sub.to_csv('../result/submission/submission.csv', index=False)

    # logging
    log.write('Number of boxes per image on average: %d\n'%check_num_boxes_per_image(df=df_sub))
    t1 = time.time()
    log.write('Post-process took %ds\n\n'%(t1 - t0))
    log.write('Submission saved to ./result/submission/submission.csv\n\n')
    log.write('============================================================\n\n')

if __name__ == '__main__':

    main()
