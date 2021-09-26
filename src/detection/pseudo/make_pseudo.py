import sys
sys.path.append('.')

# stblib
import os
import time
import argparse

# numlib
import numpy as np
import pandas as pd

# custom
from utils.file import Logger
from post_processing.postprocess import str2boxes_df, boxes2str_df, check_num_boxes_per_image, extract_none_probs, ensemble_image

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('-paths', type=str, nargs='+', help='paths to predicted csv', required=True)# img-level paths
    parser.add_argument('-ws', type=float, nargs='+', help='image-level ensemble weights', required=True)# img-level weights
    parser.add_argument('--iou-thr', type=float, default=0.6, help='boxes fusion iou threshold')# iou thres
    parser.add_argument('--conf-thr', type=float, default=0.0001, help='boxes fusion skip box threshold')# conf thes
    parser.add_argument('--none-thr', type=float, default=0.6, help='theshold for hard-labeling images as none-class')
    parser.add_argument('--opacity-thr', type=float, default=0.095, help='threshold for hard-labeling images as opacity-class')

    return parser.parse_args()

def main():
#if __name__ == '__main__':
    t0 = time.time()
    
    opt = parse_opt()

    assert len(opt.paths) == len(opt.ws), f'len(paths) == {len(ws)}'
    
    # logging
    log = Logger()
    os.makedirs('../logging', exist_ok=True)
    log.open(os.path.join('../logging', 'hard_pseudo_label.txt'), mode='a')
    log.write('weight\tpath\n')
    for p, w in zip(opt.paths, opt.ws):
        log.write('%.2f\t%s\n'%(w, p))
    log.write('\n')
    log.write('iou_thr=%.4f,skip_box_thr=%.4f\n'%(opt.iou_thr, opt.conf_thr))
    log.write('none_thr=%.4f,opacity_thre=%.4f\n'%(opt.none_thr, opt.opacity_thr))

    # prepare data    
    dfs = [pd.read_csv(df_path) for df_path in opt.paths]
    df_meta = pd.read_csv('../dataset/image-level-psuedo-label-metadata-siim/bimcv_ricord.csv')

    # post-process
    df_image = ensemble_image(dfs, df_meta, mode='wbf', \
            iou_thr=opt.iou_thr, skip_box_thr=opt.conf_thr, weights=opt.ws, filter_rows=False)[0]

    # hard-label
    bboxes = str2boxes_df(df_image, with_none=False)
    none_probs = extract_none_probs(bboxes)
    none_idx = np.array(none_probs) > opt.none_thr
    updated_boxes = [[box for box in bboxes[i] if box[0] > opt.opacity_thr] for i in range(len(bboxes)) if i not in np.where(none_idx)[0]]
    update_boxes_str = boxes2str_df(updated_boxes, image_ids=None)
    df_image.loc[none_idx, 'image_label'] = 'none 0 0 1 1'
    df_image.loc[~none_idx, 'image_label'] = update_boxes_str
    df_image = df_image[df_image['image_label'].notnull()]

    os.makedirs('../result/pseudo/hard_label', exist_ok=True)
    save_path = os.path.abspath(increment_path('../result/pseudo/hard_label/hard_label.csv', exist_ok=False))
    df_image.to_csv(, index=False)

    # logging
    log.write('Number of boxes per image on average: %d\n'%check_num_boxes_per_image(df=df_image, filter_rows=False))
    t1 = time.time()
    log.write('Hard-labeling tooks %ds\n\n'%(t1 - t0))
    log.write(f'Result is saved at {save_path}\n\n')
    log.write('============================================================\n\n')

if __name__ == '__main__':

    main()
