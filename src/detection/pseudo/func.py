import os
import shutil
from glob import glob
from tqdm.auto import tqdm
import gc

import numpy as np
import pandas as pd
import cv2

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
    df = pd.read_csv('../dataset/image-level-psuedo-label-metadata-siim/bimcv_ricord.csv')[:2]
    for _, row in tqdm(df.iterrows(), total=len(df)):
        shutil.copy(row['filepath'], save_dir)
