import sys
sys.path.append('.')

# stdlib
import os
import shutil
from glob import glob
from tqdm.auto import tqdm
import pickle
import time
import json
from collections import defaultdict

# numerical lib
import numpy as np
import pandas as pd

class Csv2Coco:
    def __init__(self, meta, with_labels=True):
        self.meta = meta
        self.with_labels = with_labels
        self.anno_id = 0

    def get_categories(self):
        categories = []
        for i, v in enumerate(Cfg.class_names):
            category = {}
            category['id'] = i
            category['name'] = v
            categories.append(category)
            
        return categories
    
    def get_images(self):
        images = []
        for i, row in self.meta.iterrows():
            image = {}
            image['width'] = Cfg.image_size
            image['height'] = Cfg.image_size
            image['id'] = row['id']
            image['file_name'] = row['id'] + '.png'
            images.append(image)

        return images
            
    def get_box(self, box):
        """
        input: [x_min y_min x_max y_max] (VOC)
        output: [x_min y_min width height] (COCO)
        element range: [0, inf/+]
        """
        assert box[0] < box[2], print(box[0], box[2])
        assert box[1] < box[3], print(box[1], box[3])

        return [box[0], box[1], box[2] - box[0], box[3] - box[1]]
        
    def get_area(self, box):
        return (box[2] - box[0]) * (box[3] - box[1])
        
    def get_annotations(self):
        annotations = []
        for i, row in self.meta.iterrows():
            if row['image_label'] == 'none 1 0 0 1 1':
                continue
            bboxes = row['image_label'].strip().split()
            bboxes = np.array([bboxes[6*idx+2:6*idx+6] for idx in range(len(bboxes)//6)]).astype('float')
            # normalize
            bboxes[..., [0, 2]] = bboxes[..., [0, 2]] * Cfg.image_size / row['w']
            bboxes[..., [1, 3]] = bboxes[..., [1, 3]] * Cfg.image_size / row['h']
            bboxes = bboxes.astype('int').tolist()

            for box in bboxes:
                annotation = {}
                annotation['id'] = self.anno_id
                annotation['image_id'] = row['id']
                annotation['category_id'] = 0
                annotation['bbox'] = self.get_box(box)
                annotation['iscrowd'] = 0
                annotation['area'] = self.get_area(box)

                annotations.append(annotation)
                self.anno_id += 1
            
        return annotations
        
    def to_coco(self):
        instance = {}
        instance['info'] = 'I love Yolo <3'
        instance['license'] = ['license']
        instance['images'] = self.get_images()
        instance['annotations'] = self.get_annotations() if self.with_labels else []
        instance['categories'] = self.get_categories()

        return instance
    
    def save_coco_json(self, filepath):
        instance = self.to_coco()
        json.dump(instance, open(filepath, 'w'), ensure_ascii=False, indent=2)

def get_df(csv_path=Cfg.csv_path, train=True):
    df = pd.read_csv(csv_path)
    if train:
        df = df[df['image_label'].notnull() * df['split']=='train'].reset_index(drop=True)
        df['filepath'] = Cfg.train_img_root + df['id'] + '.png'
        return df
    else:
        df = df[df['split']=='test'].reset_index(drop=True)
        df['filepath'] = Cfg.test_img_root + df['id'] + '.png'
        return df

def make_fold(mode='train-0'):
    if 'train' in mode:
        df = get_df(train=True)
        df_fold = pd.read_csv(os.path.join(INPUT_BASE, '4-june-2021/2021-06-14/data/siim-covid19-detection/df_fold_rand830.csv'))
        
        df = pd.merge(df_fold, df, left_on='study_id', right_on='id1', how='right')
        duplicate = read_list_from_file('../dataset/4-june-2021/2021-06-14/data/siim-covid19-detection/duplicate.txt')
        df = df[~df['id1'].isin(duplicate)]

        #---
        fold = int(mode[-1])
        df_train = df[df['fold'] != fold].reset_index(drop=True)
        df_valid = df[df['fold'] == fold].reset_index(drop=True)
        
        if Cfg.debug:
            df_train = df_train[:15]
            df_valid = df_valid[:15]

        return df_train, df_valid

def allocate_files(fold, is_train=True):
    anno_dir = os.path.join(Cfg.save_coco_path, 'annotations')
    train_dir = os.path.join(Cfg.save_coco_path, 'images', 'train2017')
    val_dir = os.path.join(Cfg.save_coco_path, 'images', 'val2017')
    test_dir = os.path.join(Cfg.save_coco_path, 'images', 'test2017')

    os.makedirs(anno_dir, exist_ok=True)
    
    if fold is not None:
        df_train, df_val = make_fold('train-%d'%fold)

        if is_train:
            #----train data
            if os.path.exists(train_dir):
                shutil.rmtree(train_dir)
            os.makedirs(train_dir, exist_ok=True)
            
            csv2coco_train = Csv2Coco(df_train, with_labels=True)
            csv2coco_train.save_coco_json(os.path.join(anno_dir, 'instances_train2017.json'))
            for i, row in tqdm(df_train.iterrows(), total=len(df_train)):
                shutil.copy(row['filepath'], os.path.join(train_dir, row['id'] + '.png'))
                
            #----valid data
            if os.path.exists(val_dir):
                shutil.rmtree(val_dir)
            os.makedirs(val_dir, exist_ok=True)
            csv2coco_val = Csv2Coco(df_val, with_labels=True)
            csv2coco_val.save_coco_json(os.path.join(anno_dir, 'instances_val2017.json'))
            for i, row in tqdm(df_val.iterrows(), total=len(df_val)):
                shutil.copy(row['filepath'], os.path.join(val_dir, row['id'] + '.png'))
        else:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
            os.makedirs(test_dir, exist_ok=True)
            
            csv2coco_val = Csv2Coco(df_val, with_labels=True)
            csv2coco_val.save_coco_json(os.path.join(anno_dir, 'instances_test2017.json'))
            for i, row in tqdm(df_val.iterrows(), total=len(df_val)):
                shutil.copy(row['filepath'], os.path.join(test_dir, row['id'] + '.png'))
    else:
        #----test data
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        os.makedirs(test_dir, exist_ok=True)
        
        df_test = make_fold('test')

        os.makedirs(anno_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        csv2coco_test = Csv2Coco(df_test, with_labels=False)
        csv2coco_test.save_coco_json(os.path.join(anno_dir, 'instances_test2017.json'))
        for i, row in tqdm(df_test.iterrows(), total=len(df_test)):
            shutil.copy(row['filepath'], os.path.join(test_dir, row['id'] + '.png'))

    if 'test' in mode:
        df_test = get_df(train=False)
        df_test = df_test.reset_index(drop=True)
        
        if Cfg.debug:
            df_test = df_test[:15]

        return df_test

def get_image_sub(pickle_file_path, df_test, image_size):
    # output [opacity prob x_min y_min x_max y_max]
    ids, prediction_strings = [], []
    preds = pickle.load(open(pickle_file_path, 'rb'))

    for i, row in tqdm(df_test.iterrows(), total=len(df_test)):
        bboxes = preds[i][0]
        bboxes = np.concatenate([[['opacity']]*bboxes.shape[0], bboxes[:,-1:], bboxes[:,:-1]], axis=1)
        bboxes[..., [2, 4]] = bboxes[..., [2, 4]].astype(np.float32) * row['w'] / image_size
        bboxes[..., [3, 5]] = bboxes[..., [3, 5]].astype(np.float32) * row['h'] / image_size
        bboxes = bboxes.reshape(-1).tolist()

        ids.append(row['id'] + '_image')
        prediction_strings.append(' '.join(bboxes))

    return pd.DataFrame({'id': ids, 'PredictionString': prediction_strings})
