import os
import torch

class Config:
    seed = 42
    num_classes = 1
    class_names  = ['opacity']

    image_size = 1024
    train_img_root = '../dataset/1024x1024-png-siim/train/'
    test_img_root = '../dataset/1024x1024-png-siim/test/'

    csv_path = '../dataset/meta.csv'
    ext = '.png'

    fold_path = '../dataset/fold-split-siim/df_fold.csv'
    duplicate_path = '../dataset/fold-split-siim/duplicate.txt'

    yaml_data_path = './detection/yolo/data/data.yaml'
    yaml_hyp_path = './detection/yolo/data/hyp.yaml'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
