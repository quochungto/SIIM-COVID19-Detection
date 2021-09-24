import sys
sys.path.append('.')

# stdlib
import os
import argparse

# numlib
import numpy as np
import pandas as pd

from ensemble_boxes import nms, weighted_boxes_fusion

# custum
from utils.common.file import Logger

def str2boxes_image(s, with_none=False):
    """
    ouput: [prob x_min y_min x_max y_max]
    range x,y: [0, +inf]
    """ 
    s = s.strip().split()
    s = np.array([s[6*idx+1:6*idx+6] for idx in range(len(s)//6) if s[6*idx] == 'opacity' or with_none]).astype(np.float32)
    if len(s) == 0: print('Warning: image without box!')
    return s


def str2boxes_df(df, with_none=False):
    return [str2boxes_image(row['PredictionString'], with_none=with_none) for _, row in df.iterrows()]


def boxes2str_image(boxes):
    if len(boxes) == 0:
        return ''
    return ' '.join(np.concatenate([[['opacity']]*len(boxes), boxes], axis=1).reshape(-1).astype('str'))


def boxes2str_df(boxes, image_ids=None):
    strs = [boxes2str_image(bs) for bs in boxes]
    if image_ids is None:
        return strs
    return pd.DataFrame({'id': image_ids, 'PredictionString': strs})


def check_num_boxes_per_image(df=None, csv_path=None):
    assert df is not None or csv_path is not None
    if df is None:
        df = pd.read_csv(csv_path)
    all_boxes = str2boxes_df(df, with_none=False)
    all_boxes = [boxes for boxes in all_boxes if len(boxes) > 0 ]
    return np.concatenate(all_boxes).shape[0] / len(df)


def extract_none_probs(opacity_probs):
    none_probs = []

    for image_probs in opacity_probs:
        none_prob = np.prod(1 - np.array(image_probs))
        none_probs.append(none_prob)

    return none_probs


def correct_ids(df, mode='image'):
    assert mode in ['study', 'image']
    df = df.copy()
    df['id'] = df['id'].apply(lambda x: x + '_' + mode if not x.endswith(mode) else x)
    return df


def ensemble_pred(df_paths,
                  df_test,
                  mode='wbf',
                  iou_thr=0.5, 
                  skip_box_thr=0.001, 
                  weights=None):
    
    df_test = correct_ids(df_test, mode='image')
    
    dfs = [correct_ids(pd.read_csv(df_path), mode='image') for df_path in df_paths]
    
    image_ids, PredictionStrings, all_scores = [], [], []
    num_boxes = 0
    
    for i, row in tqdm(df_test.iterrows(), total=len(df_test)):
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
            PredictionStrings.append('')
            print('Warning: no box found after boxes fusion!')
            continue

        num_boxes += len(boxes)
        all_scores.append(scores)

        boxes = upsize_boxes(boxes, row['w'], row['h'])
        
        s = []
        for box, score, label in zip(boxes, scores, labels):
            s.append(' '.join(['opacity', str(score), ' '.join(box.astype(str))]))

        image_ids.append(image_id)
        PredictionStrings.append(' '.join(s))

    df_pred = pd.DataFrame({'id': image_ids, 'PredictionString': PredictionStrings})

    return df_pred, num_boxes, np.concatenate(all_scores).tolist()

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
    os.makedirs('./logging', exist_ok=True)
    log.open(os.path.join('./logging', 'post_processing.txt'), mode='a')
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
