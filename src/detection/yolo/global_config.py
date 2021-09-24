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

    fold_path = '../dataset/4-june-2021/2021-06-14/data/siim-covid19-detection/df_fold_rand830.csv'
    duplicate_path = '../dataset/4-june-2021/2021-06-14/data/siim-covid19-detection/duplicate.txt'

    yaml_data_path = './detection/yolo/data.yaml'
    yaml_hyp_path = './detection/yolo/hyp.yaml'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
