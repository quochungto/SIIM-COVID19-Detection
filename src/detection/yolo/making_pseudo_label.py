import os
os.environ['WANDB_MODE'] = 'dryrun'
import sys
import shutil
import yaml
import json
from collections import defaultdict
from glob import glob
from tqdm.notebook import tqdm
import gc
import re
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import torch

class Config:
    seed = 42
    num_classes = 1
    class_names = ['opacity']
    train_count = 0

    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    iou_detect = 0.5
    conf_detect = 0.001

    # debug
    debug = False

def get_df():
    ricord_sample_paths = glob('../dataset/ricord-covid19-xray-positive-tests/MIDRC-RICORD/MIDRC-RICORD/*.jpg')
    ricord_ids = ['.'.join(os.path.split(path)[-1].split('.')[:-1]) for path in ricord_sample_paths]
    ricord_hw = [cv2.imread(path).shape[:2] for path in ricord_sample_paths]
    ricord_h = [shape[0] for shape in ricord_hw]
    ricord_w = [shape[1] for shape in ricord_hw]
    pd.DataFrame({'filepath': ricord_sample_paths, 'w': ricord_w, 'h': ricord_h})
    df_ricord = pd.DataFrame({'id': ricord_ids, 'filepath': ricord_sample_paths, 'w': ricord_w, 'h': ricord_h})

    bimcv_sample_names = np.load('../dataset/bimcv-invalid-image-names-siim/valid_images.npy')
    bimcv_ids = ['.'.join(name.split('.')[:-1]) for name in bimcv_sample_names]
    bimcv_sample_paths = [os.path.join('/kaggle/input/covid19-posi-dump-siim/covid19_posi_dump', name) for name in bimcv_sample_names]
    df_bimcv = pd.DataFrame({'id': bimcv_ids, 'filepath': bimcv_sample_paths, 'w': len(bimcv_sample_paths)*[512], 'h': len(bimcv_sample_paths)*[512]})

    return pd.concat([df_ricord, df_bimcv], ignore_index=True)

def allocate_pseudo_files(save_dir):
    if os.path.exists(save_dir): 
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    #df = get_df()
    df = pd.read_csv('../dataset/image-level-psuedo-label-metadata-siim/bimcv_ricord.csv')
    for _, row in tqdm(df.iterrows(), total=len(df)):
        shutil.copy(row['filepath'], save_dir)

# checkpoint name format: modelname - fold - input size - batch size - epoch
def pseudo_predict(ck_path,
               image_size,
               save_dir=os.path.join(WORKING_BASE, 'yolo_pseudo'),
               batch_size=128,
               iou_thresh=0.5,
               conf_thresh=0.001,):
    memory_cleanup()

    ck_name = os.path.split(ck_path)[-1].split('.')[0]
    sname = re.sub('[^\w_-]', '', 'iou%.2f_conf%.4f'%(iou_thresh, conf_thresh))
    save_dir = os.path.join(save_dir, sname, ck_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f'Running {ck_name} - {sname}'.upper())
    print('\n')

    df_valid = pd.read_csv('../dataset/image-level-psuedo-label-metadata-siim/bimcv_ricord.csv')
    test_image_dir = os.path.join(DATA_BASE, 'chest')
    allocate_pseudo_files(test_image_dir)

    exp_path = './detection/yolov5/runs/detect/exp'
    if os.path.exists(exp_path): shutil.rmtree(exp_path)

    os.chdir(os.path.join(WORKING_BASE, 'yolov5'))

    !python ./detect.py \
    --weights {ck_path} \
    --img {image_size} \
    --conf {conf_thresh} \
    --iou {iou_thresh} \
    --source {test_image_dir} \
    --augment \
    --save-txt \
    --save-conf \
    --exist-ok \
    --nosave \

    prediction_path = os.path.join(exp_path, 'labels')
    df_sub = yolo_get_image_sub(prediction_path, df_valid)
    df_sub.to_csv(os.path.join(exp_path, 'image_sub.csv'), index=False)
    os.makedirs('/content/pseudo_output', exist_ok=True)
    df_sub.to_csv(os.path.join('/content/pseudo_output', f'{ck_name}_{sname}.csv'), index=False)
    df_valid.to_csv(os.path.join(exp_path, 'valid.csv'), index=False)

    return shutil.move(exp_path, save_dir)
    #return save_checkpoints(exp_path, save_dir, prefix=None, mode='move')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ck-paths', type=str, nargs='+')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--conf-thr', type=float, default=0.001)
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--redownload', action='store_true')
    parser.add_argument('--device', default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    return parser.parse_args()

def main():
    seed_everything(Config.seed)
    Config.debug = False

#    df = pd.read_csv('../dataset/image-level-psuedo-label-metadata-siim/bimcv_ricord.csv')
#    df['filepath'] = df['filepath'].apply(lambda x: x.replace('/kaggle/input', '../dataset'))
#    df.to_csv('../dataset/image-level-psuedo-label-metadata-siim/bimcv_ricord.csv')

    opt = parse_opt()
    
    # parser
    ck_paths, image_size, batch_size, \
    iou, conf, mode, debug, \
    redownload, device = \
    opt.ck_paths, opt.image_size, opt.batch_size, \
    opt.iou_thr, opt.conf_thr, opt.mode, opt.debug, \
    opt.redownload, opt.device
    
#    batch_size = 128
#    iou_thresh = 0.5
#    conf_thresh = 0.001
#    image_size = 640

#    if WORKING_BASE == '/kaggle/working':
#	save_dir = os.path.join(WORKING_BASE, 'yolo_pseudo')
#    else:
#	save_dir = '/content/drive/My Drive/siim/yolo_pseudo'

    save_dir = '../result/pseudo'

#    ck_paths = glob('/content/kaggle_datasets/checkpoints-yolo5x-512-8-35-roccfg-siim/*')

    for ck_path in tqdm(ck_paths):
	saved_dir = pseudo_predict(ck_path, image_size,
		   save_dir=save_dir,
		   batch_size=batch_size,
		   iou_thresh=iou,
		   conf_thresh=conf)
	print('Results saved to %s'%saved_dir)
	print('\n====================================================================\n')

if __name__ == '__main__':
    main()
