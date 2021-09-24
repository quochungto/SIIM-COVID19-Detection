import sys
sys.path.extend([#'/kaggle/input/weightedboxfusion',
                 '/kaggle/input/dummy-siim',
                 '/kaggle/input/dummy0'])
import os
import shutil
from tqdm.notebook import tqdm
from glob import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from hung_common import *
from him import *
from post_processing import *

def main():
    df_test = pd.read_csv('../input/original-size-png-public-test-images-siim/test.csv')
    df_test['id'] = df_test['id'] + '_image'

    df_paths = []

    ws = None
    df_test.tail()

    for df_path in df_paths:
	s = '%s: %d bboxes per image'%(os.path.split(df_path)[-1].split('.')[0], check_num_boxes_per_image(csv_path=df_path))
	print(s.upper(), end='\n\n')

    file_name = 'oldyolov5x640_yolotrs384_yolov5l6512_yolov5x512pseudo_iou06_weights2223.csv'
    df_pred, num_boxes, scores = ensemble_pred(df_paths,
						df_test,
						mode='wbf',
						iou_thr=0.6,
						skip_box_thr=0.001,
						weights=[2,2,2,2,2,
							 2,2,2,2,2,
							 2,2,2,2,2,
							 3,3,3,3,3,
							])

    df_pred.to_csv(file_name, index=False)
    print(f'num_boxes_per_image = {num_boxes / 1263:.2f} | file_size = {os.stat(file_name).st_size/(2**20):.2f} MB')
    # df_pred.tail()

if __name__ == '__main__':
    main()
