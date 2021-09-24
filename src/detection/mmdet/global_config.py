import torch

class Cfg:
    seed = 42
    num_classes = 1
    class_names  = ['opacity']

    cfg_path = '../dataset/mmdetection-configs-siim/vfnet_cfg.py')
    ckpt_path = '../result/mmdet/checkpoints'
    
    image_size = 1024
    train_img_root = '../dataset/1024x1024-png-siim/train/'
    test_img_root = '../dataset/1024x1024-png-siim/test/'
    csv_path = '../dataset/1024x1024-png-siim/meta.csv'
    ext = '.png'

    test_image_size = None

    input_size = 640
    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    iou_detect = 0.5
    conf_detect = 0.001
  
    debug = False
    
    save_coco_path = '../dataset/coco'
