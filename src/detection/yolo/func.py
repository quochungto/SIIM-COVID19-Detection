import sys
sys.path.append('.')

# stdlib
import os
import shutil
import time
import yaml
from tqdm.auto import tqdm

# numerical lib
import numpy as np
import pandas as pd

from global_config import Config
from utils.file import Logger, read_list_from_file
from utils.him import downsize_boxes, upsize_boxes, voc2yolo
from utils.torch_common import memory_cleanup
#from utils.siim.yolo import allocate_files

# functions
def get_df(csv_path, train=True):
    df = pd.read_csv(csv_path)
    if train:
        df = df[df['image_label'].notnull() * df['split']=='train'].reset_index(drop=True)
        return df
    else:
        df = df[df['split']=='test'].reset_index(drop=True)
        return df


def make_fold(mode, csv_path, fold_path=None, duplicate_path=None, pseudo_csv_path=None):
    if 'train' in mode:
        df = get_df(csv_path, train=True)
        df_fold = pd.read_csv(fold_path)

        df = pd.merge(df_fold, df, left_on='study_id', right_on='id1', how='right')

        if duplicate_path is not None:
            duplicate = read_list_from_file(duplicate_path)
            df = df[~df['id1'].isin(duplicate)]

        #---
        fold = int(mode[-1])
        df_train = df[df['fold'] != fold].reset_index(drop=True)
        df_valid = df[df['fold'] == fold].reset_index(drop=True)

        if pseudo_csv_path is not None:
            df_pseudo = pd.read_csv(pseudo_csv_path)
            df_train = pd.concat([df_train, df_pseudo], axis=0)
            df_train = df_train.reset_index(drop=True)

        return df_train, df_valid

    if 'test' in mode:
        df_test = get_df(csv_path, train=False)
        df_test = df_test.reset_index(drop=True)

        return df_test


def get_yolo_bboxes_from_input_df(row):
    # voc2yolo
    if row['image_label'] == 'none 1 0 0 1 1':
        return np.array([])
    bboxes = row['image_label'].strip().split()
    bboxes = np.array([bboxes[6*idx+2:6*idx+6] for idx in range(len(bboxes)//6)], dtype=np.float32)
    bboxes = downsize_boxes(bboxes, row['w'], row['h'])
    bboxes = voc2yolo(bboxes)
    bboxes = np.concatenate([[['0']]*len(bboxes), bboxes], axis=1)

    return bboxes


def write_annotations(df, image_dir, label_dir):
    for _, row in tqdm(df.iterrows(), total=len(df)):
        #----copy images
        shutil.copy(row['filepath'], image_dir)
        #----

        #----write labels
        if label_dir is not None:
            bboxes = get_yolo_bboxes_from_input_df(row)
            s = [' '.join(box) for box in bboxes.astype('str')]
            s = '\n'.join(s)
            with open(os.path.join(label_dir, row['id'] + '.txt'), 'w') as out:
                out.write(s)
                #out.write('\n')
        #----


def allocate_files(fold,
                csv_path,
                yaml_path,
                save_dir,
                num_classes,
                class_names,
                is_train=True,
                fold_path=None,
                duplicate_path=None,
                pseudo_csv_path=None):
    if fold is not None:
        train_image_dir = os.path.join(save_dir, 'images', 'train')
        train_label_dir = train_image_dir.replace('images', 'labels')
        val_image_dir = train_image_dir.replace('train', 'val')
        val_label_dir = train_label_dir.replace('train', 'val')

        if os.path.exists(train_image_dir): shutil.rmtree(train_image_dir)
        if os.path.exists(train_label_dir): shutil.rmtree(train_label_dir)
        if os.path.exists(val_image_dir): shutil.rmtree(val_image_dir)
        if os.path.exists(val_label_dir): shutil.rmtree(val_label_dir)

        os.makedirs(train_image_dir, exist_ok=True)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(val_image_dir, exist_ok=True)
        os.makedirs(val_label_dir, exist_ok=True)

        df_train, df_valid = make_fold('train-%d'%fold, csv_path, fold_path, duplicate_path, pseudo_csv_path)
        df_train = df_train[:15]
        df_valid = df_valid[:15]

        if is_train: write_annotations(df_train, train_image_dir, train_label_dir)
        write_annotations(df_valid, val_image_dir, val_label_dir)

        config = {'train': train_image_dir,
                 'val': val_image_dir,
                 'nc': num_classes,
                 'names': class_names}

        with open(yaml_path, 'w') as out:
            yaml.dump(config, out, default_flow_style=False)

        return train_image_dir, train_label_dir, val_image_dir, val_label_dir

    else:
        test_image_dir = os.path.join(save_dir, 'images', 'test')

        if os.path.exists(test_image_dir): shutil.rmtree(test_image_dir)
        os.makedirs(test_image_dir, exist_ok=True)

        df_test = make_fold('test', csv_path)
        df_test = df_test[:15]

        write_annotations(df_test, test_image_dir, None)

        return test_image_dir


def get_image_sub(prediction_path, df_test):
    # output [opacity prob x_min y_min x_max y_max]
    image_ids, PredictionStrings = [], []

    for txt_path in tqdm(glob(os.path.join(prediction_path, '*.txt'))):

        image_id = '.'.join(os.path.split(txt_path)[-1].split('.')[:-1])
        image_ids.append(image_id + '_image')
        w, h = df_test.loc[df_test['id']==image_id, ['w', 'h']].values[0]

        with open(txt_path, 'r') as f:
            bboxes = np.array(f.read().replace('\n', ' ').strip().split()).astype(np.float32).reshape(-1, 6)
            bboxes = bboxes[:, [0, 5, 1, 2, 3, 4]] # [class_id prob x_cen y_cen width height]
            boxes = upsize_boxes(yolo2voc(bboxes[:, 2:]), w, h)
            bboxes = np.concatenate([bboxes[:, :2], boxes], axis=1).astype('str')
            bboxes[:, 0] = 'opacity'
            bboxes = bboxes.reshape(-1)

            PredictionStrings.append(' '.join(bboxes))

    return pd.DataFrame({'id': image_ids, 'PredictionString': PredictionStrings})


