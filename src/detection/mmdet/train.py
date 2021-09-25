import sys
sys.path.append('.')

# stdlib
import os
import argparse

# custom
from global_config import Cfg
from utils.torch_common import seed_everything, 
from func import allocate_files

# mmdetection
import mmdet
import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, set_random_seed

def train(folds, epochs=25, cfg_path=Cfg.cfg_path):
    for fold in folds:
        memory_cleanup()

        print(f'---- Training fold {fold} ----')
        allocate_files(fold, is_train=True)

        cfg = Config.fromfile(cfg_path)
        #--adapt config for each platform
        #----for both
        cfg.work_dir = './exps/fold%d'%fold
        cfg.total_epochs = epochs
        cfg.runner['max_epochs'] = epochs
        #----for google colab
        #if WORKING_BASE == '/content':

        #    cfg.data_root = cfg.data_root.replace('/kaggle/tmp', DATA_BASE)
        #    cfg.data.train.data_root = cfg.data.train.data_root.replace('/kaggle/tmp', DATA_BASE)
        #    cfg.data.val.data_root = cfg.data.val.data_root.replace('/kaggle/tmp', DATA_BASE)
        #    cfg.data.test.data_root = cfg.data.test.data_root.replace('/kaggle/tmp', DATA_BASE)
        #    cfg.load_from = cfg.load_from.replace('/kaggle/input', INPUT_BASE)            

        datasets = build_dataset([cfg.data.train])
        model = build_detector(cfg.model)
        train_detector(model, datasets, cfg, validate=True, distributed=False)
        print('\n')
    if BACKEND == 'colab':
      #return save_checkpoints(os.path.join(WORKING_BASE, 'exps'), '/content/working/drive/siim',prefix='mmdet_detectoRS')
      return save_checkpoints('exps', '../result/mmdet/checkpoints', prefix='mmdet_detectoRS')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', type=str, default='')
    parser.add_argument('--folds', type=str, nargs='+', default=[0, 1, 2, 3, 4], help='folds for training, i.e. 0 or 0,1,2,3,4')
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--pseudo', action='store_true')
    parser.add_argument('--device', default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    return parser.parse_args()

def main():
    seed_everything(Cfg.seed)

    #cfg_path = os.path.join(INPUT_BASE, 'mmdetection-configs-siim', 'vfnetr101_cfg.py')

    opt = parse_opt()
    
    train(opt.folds, epochs=opt.epochs, cfg_path=cfg_path)

if __name__ == '__main__':
    main()